require 'astrogemini/version'
require 'httparty'
require 'open-uri'
require 'json'
require 'mime/types'
require 'tempfile'
require 'securerandom'

module AstroGemini
  class Client
    include HTTParty

    BASE_URI = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash'
    UPLOAD_URI = 'https://generativelanguage.googleapis.com/upload/v1beta/files'
    MAX_INLINE_SIZE = 20 * 1024 * 1024 # 20MB in bytes

    def initialize(api_key)
      @api_key = api_key
      @context_array = []
    end

    def add_to_context(sources)
      sources.each do |source|
        if source.start_with?('http://', 'https://')
          @context_array << process_remote_file(source)
        elsif File.exist?(source)
          @context_array << process_local_file(source)
        else
          raise Error, "Invalid source: #{source}. It must be a URL or a file path."
        end
      end
    end

    def generate_text(given_prompt, options = {})
      full_prompt = given_prompt.to_s

      body = {
        contents: [
          {
            parts: [
              { text: full_prompt }
            ].concat(@context_array)
          }
        ]
      }.merge(options)

      response = self.class.post(
        "#{BASE_URI}:generateContent?key=#{@api_key}",
        headers: { 'Content-Type' => 'application/json' },
        body: body.to_json
      )
      handle_response(response)
    end

    def generate_embedding(text)
      body = {
        model: 'models/text-embedding-004',
        content: {
          parts: [
            { text: text.to_s }
          ]
        }
      }

      response = self.class.post(
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=#{@api_key}",
        headers: { 'Content-Type' => 'application/json' },
        body: body.to_json
      )
      handle_response(response)
    end

    private

    def handle_response(response)
      case response.code
      when 200..299
        JSON.parse(response.body)
      when 429
        raise RateLimitError, "Gemini API Rate Limit Exceeded: #{response.body}"
      else
        raise Error, "Gemini API Error: #{response.code} - #{response.body}"
      end
    end

    def process_remote_file(url)
      temp_file = download_file(url)
      process_file(temp_file, url)
    ensure
      temp_file&.close
      temp_file&.unlink
    end

    def process_local_file(file_path)
      process_file(File.new(file_path), file_path)
    end

    def process_file(file, source_identifier)
      file_size = File.size(file.path)
      mime_type = get_mime_type(file.path)

      if file_size <= MAX_INLINE_SIZE
        handle_small_file(file, mime_type, source_identifier)
      else
        handle_large_file(file, mime_type, source_identifier)
      end
    end

    def handle_small_file(file, mime_type, source_identifier)
      file.binmode
      encoded_content = Base64.strict_encode64(file.read)
      { inline_data: { mime_type: mime_type, data: encoded_content } }
    end

    def handle_large_file(file, mime_type, source_identifier)
      file_uri = upload_large_file(file, mime_type)
      { file_data: { mime_type: mime_type, file_uri: file_uri } }
    end

    def download_file(url)
      temp_file = Tempfile.new(['astrogemini', File.extname(url)], binmode: true)
      URI.open(url, 'rb') do |remote_file|
        temp_file.binmode
        temp_file.write(remote_file.read)
      end
      temp_file.rewind
      temp_file
    rescue OpenURI::HTTPError => e
      raise Error, "Failed to download file from #{url}: #{e.message}"
    end

    def upload_large_file(file, mime_type)
      file_size = File.size(file.path)
      display_name = File.basename(file.path)

      # Initiate resumable upload
      upload_url = initiate_resumable_upload(file_size, mime_type, display_name)

      # Upload file content
      upload_file_content(upload_url, file, file_size)
    end

    def initiate_resumable_upload(file_size, mime_type, display_name)
      response = self.class.post(
        "#{UPLOAD_URI}?key=#{@api_key}",
        headers: {
          'X-Goog-Upload-Protocol' => 'resumable',
          'X-Goog-Upload-Command' => 'start',
          'X-Goog-Upload-Header-Content-Length' => file_size.to_s,
          'X-Goog-Upload-Header-Content-Type' => mime_type,
          'Content-Type' => 'application/json'
        },
        body: { file: { display_name: display_name } }.to_json
      )

      upload_url = response.headers['x-goog-upload-url']
      raise Error, "Failed to initiate upload: #{response.body}" if upload_url.nil? || upload_url.strip.empty?

      upload_url
    end

    def upload_file_content(upload_url, file, file_size)
      raise Error, 'Invalid upload URL' if upload_url.nil? || upload_url.strip.empty?

      uri = URI.parse(upload_url)
      file.binmode
      response = self.class.put(
        uri.to_s,
        headers: {
          'Content-Length' => file_size.to_s,
          'X-Goog-Upload-Offset' => '0',
          'X-Goog-Upload-Command' => 'upload, finalize'
        },
        body: file.read
      )

      raise Error, "Failed to upload file content: #{response.body}" unless response.success?

      JSON.parse(response.body).dig('file', 'uri') || raise(Error, 'No file URI')
    end

    def get_mime_type(file_path)
      MIME::Types.type_for(file_path).first&.to_s || 'application/octet-stream'
    end
  end

  class Error < StandardError; end
  class RateLimitError < Error; end
end
