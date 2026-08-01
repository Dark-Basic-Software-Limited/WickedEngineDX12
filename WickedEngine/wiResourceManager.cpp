#include "wiResourceManager.h"
#include "wiRenderer.h"
#include "wiHelper.h"
#include "wiTextureHelper.h"
#include "wiUnorderedMap.h"
#include "wiBacklog.h"
#include "wiJobSystem.h"

#include "Utility/stb_image.h"
#include "Utility/dds.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <fstream> // GGMAX 1.52b: resource_hijack.txt tripwire
#include <set>     // GGMAX 1.73: streaming guard de-duplicates its report by resource+reason

using namespace wi::graphics;

//#define RESOURCE_LOGGING

#ifdef RESOURCE_LOGGING
#define resource_log(str,...) wilog(str, ## __VA_ARGS__)
#else
#define resource_log(str,...)
#endif // RESOURCE_LOGGING

namespace wi
{
	struct StreamingTexture
	{
		struct StreamingSubresourceData
		{
			size_t data_offset = 0;
			uint32_t row_pitch = 0;
			uint32_t slice_pitch = 0;
		};
		StreamingSubresourceData streaming_data[16] = {};
		uint32_t mip_count = 0; // mip count of full resource
		float min_lod_clamp_absolute = 0; // relative to mip_count of full resource
	};
	//static constexpr size_t streaming_texture_min_size = 4096; // 4KB is the minimum texture memory alignment
	static constexpr size_t streaming_texture_min_size = 64 * 1024; // 64KB is the usual texture memory alignment, this allows higher base tex size than 4KB

	struct ResourceInternal
	{
		resourcemanager::Flags flags = resourcemanager::Flags::NONE;
		wi::graphics::Texture texture;
		int srgb_subresource = -1;
		wi::audio::Sound sound;
		std::string script;
		size_t script_hash = 0;
		wi::video::Video video;
		wi::vector<uint8_t> filedata;
		int font_style = -1;

		// Original filename:
		std::string filename;

		// Container file is different from original filename when
		//	multiple resources are embedded inside one file:
		std::string container_filename;
		size_t container_filesize = ~0ull;
		size_t container_fileoffset = 0;
		uint64_t timestamp = 0;

		// Streaming parameters:
		StreamingTexture streaming_texture;
		std::atomic<uint32_t> streaming_resolution{ 0 };
		uint32_t streaming_unload_delay = 0;

		// Virtual texture things:
		wi::graphics::GPUBuffer tile_pool;
		wi::graphics::Texture texture_feedback;
		wi::graphics::Texture texture_residency;
	};

	const wi::vector<uint8_t>& Resource::GetFileData() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->filedata;
	}
	const wi::graphics::Texture& Resource::GetTexture() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->texture;
	}
	const wi::audio::Sound& Resource::GetSound() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->sound;
	}
	const std::string& Resource::GetScript() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->script;
	}
	size_t Resource::GetScriptHash() const
	{
		if (internal_state == nullptr)
			return 0;
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->script_hash;
	}
	const wi::video::Video& Resource::GetVideo() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->video;
	}
	int Resource::GetTextureSRGBSubresource() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->srgb_subresource;
	}
	int Resource::GetFontStyle() const
	{
		const ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		return resourceinternal->font_style;
	}

	void Resource::SetFileData(const wi::vector<uint8_t>& data)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->filedata = data;
	}
	void Resource::SetFileData(wi::vector<uint8_t>&& data)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->filedata = data;
	}
	void Resource::SetTexture(const wi::graphics::Texture& texture, int srgb_subresource)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->texture = texture;
		resourceinternal->srgb_subresource = srgb_subresource;
	}
	void Resource::SetTextureVirtual(const GPUBuffer& tile_pool, const Texture& residency, const Texture& feedback)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->tile_pool = tile_pool;
		resourceinternal->texture_residency = residency;
		resourceinternal->texture_feedback = feedback;
	}
	void Resource::SetSound(const wi::audio::Sound& sound)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->sound = sound;
	}
	void Resource::SetScript(const std::string& script)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->script = script;
	}
	void Resource::SetVideo(const wi::video::Video& video)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->video = video;
	}

	void Resource::SetOutdated()
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		resourceinternal->timestamp = 0;
	}

	// GGMAX 1.69: feedback-chain probe — nonzero resolution requests reaching resources
	std::atomic<unsigned long long> gg_dbg_stream_req_calls{ 0 };

	void Resource::StreamingRequestResolution(uint32_t resolution)
	{
		if (internal_state == nullptr)
		{
			internal_state = wi::allocator::make_shared<ResourceInternal>();
		}
		ResourceInternal* resourceinternal = (ResourceInternal*)internal_state.get();
		if (resolution != 0)
		{
			gg_dbg_stream_req_calls.fetch_add(1, std::memory_order_relaxed); // GGMAX 1.69
		}
		resourceinternal->streaming_resolution.fetch_or(resolution);
	}

	namespace resourcemanager
	{
		static std::mutex locker;
		static wi::unordered_map<std::string, wi::allocator::weak_ptr<ResourceInternal>> resources;
		static Mode mode = Mode::NO_EMBEDDING;

		// GGMAX 1.41: bumped whenever texture streaming swaps a texture object or recreates
		// subresources — any cached GPU descriptor index derived from a streamed resource is
		// invalid across a bump. Consumed by the ShaderMaterial recompose cache in wiScene.
		std::atomic<uint32_t> gg_streaming_descriptor_epoch{ 0 };

		// GGMAX 1.73 DIAG: see the definition comment further down — declared here because
		// LoadResourceDirectly (above the streaming machinery) is the site that writes the trace.
		extern bool gg_stream_load_trace;

		void SetMode(Mode param)
		{
			mode = param;
		}
		Mode GetMode()
		{
			return mode;
		}

		enum class DataType
		{
			IMAGE,
			SOUND,
			SCRIPT,
			VIDEO_MP4,
			VIDEO_H264_RAW,
			FONTSTYLE,
		};
		static const wi::unordered_map<std::string, DataType> types = {
			{"JPG", DataType::IMAGE},
			{"JPEG", DataType::IMAGE},
			{"PNG", DataType::IMAGE},
			{"BMP", DataType::IMAGE},
			{"DDS", DataType::IMAGE},
			{"TGA", DataType::IMAGE},
			{"HDR", DataType::IMAGE},
			{"WAV", DataType::SOUND},
			{"OGG", DataType::SOUND},
			{"LUA", DataType::SCRIPT},
			{"MP4", DataType::VIDEO_MP4},
			{"H264", DataType::VIDEO_H264_RAW},
			{"TTF", DataType::FONTSTYLE},
		};
		wi::vector<std::string> GetSupportedImageExtensions()
		{
			wi::vector<std::string> ret;
			for (auto& x : types)
			{
				if (x.second == DataType::IMAGE)
				{
					ret.push_back(x.first);
				}
			}
			return ret;
		}
		wi::vector<std::string> GetSupportedSoundExtensions()
		{
			wi::vector<std::string> ret;
			for (auto& x : types)
			{
				if (x.second == DataType::SOUND)
				{
					ret.push_back(x.first);
				}
			}
			return ret;
		}
		wi::vector<std::string> GetSupportedVideoExtensions()
		{
			wi::vector<std::string> ret;
			for (auto& x : types)
			{
				if (x.second == DataType::VIDEO_MP4 || x.second == DataType::VIDEO_H264_RAW)
				{
					ret.push_back(x.first);
				}
			}
			return ret;
		}
		wi::vector<std::string> GetSupportedScriptExtensions()
		{
			wi::vector<std::string> ret;
			for (auto& x : types)
			{
				if (x.second == DataType::SCRIPT)
				{
					ret.push_back(x.first);
				}
			}
			return ret;
		}
		wi::vector<std::string> GetSupportedFontStyleExtensions()
		{
			wi::vector<std::string> ret;
			for (auto& x : types)
			{
				if (x.second == DataType::FONTSTYLE)
				{
					ret.push_back(x.first);
				}
			}
			return ret;
		}

		bool LoadResourceDirectly(
			const std::string& name,
			Flags flags,
			const uint8_t* filedata,
			size_t filesize,
			ResourceInternal* resource
		)
		{
			std::string ext = wi::helper::toUpper(wi::helper::GetExtensionFromFileName(name));
			DataType type;

			// dynamic type selection:
			{
				auto it = types.find(ext);
				if (it != types.end())
				{
					type = it->second;
				}
				else
				{
					return false;
				}
			}

			bool success = false;

			switch (type)
			{
			case DataType::IMAGE:
			{
				GraphicsDevice* device = wi::graphics::GetDevice();
				if (!ext.compare("DDS"))
				{
					dds::Header header = dds::read_header(filedata, filesize);
					if (header.is_valid())
					{
						TextureDesc desc;
						desc.array_size = 1;
						desc.bind_flags = BindFlag::SHADER_RESOURCE;
						desc.width = header.width();
						desc.height = header.height();
						desc.depth = header.depth();
						desc.mip_levels = header.mip_levels();
						desc.array_size = header.array_size();
						desc.format = Format::R8G8B8A8_UNORM;
						desc.layout = ResourceState::SHADER_RESOURCE;
						desc.misc_flags = ResourceMiscFlag::TYPED_FORMAT_CASTING;

						if (header.is_cubemap())
						{
							desc.misc_flags |= ResourceMiscFlag::TEXTURECUBE;
						}
						if (desc.mip_levels == 1 || desc.depth > 1 || desc.array_size > 1)
						{
							// don't allow streaming for single mip, array and 3D textures
							flags &= ~Flags::STREAMING;
						}

						auto ddsFormat = header.format();

						switch (ddsFormat)
						{
						case dds::DXGI_FORMAT_R32G32B32A32_FLOAT: desc.format = Format::R32G32B32A32_FLOAT; break;
						case dds::DXGI_FORMAT_R32G32B32A32_UINT: desc.format = Format::R32G32B32A32_UINT; break;
						case dds::DXGI_FORMAT_R32G32B32A32_SINT: desc.format = Format::R32G32B32A32_SINT; break;
						case dds::DXGI_FORMAT_R32G32B32_FLOAT: desc.format = Format::R32G32B32_FLOAT; break;
						case dds::DXGI_FORMAT_R32G32B32_UINT: desc.format = Format::R32G32B32_UINT; break;
						case dds::DXGI_FORMAT_R32G32B32_SINT: desc.format = Format::R32G32B32_SINT; break;
						case dds::DXGI_FORMAT_R16G16B16A16_FLOAT: desc.format = Format::R16G16B16A16_FLOAT; break;
						case dds::DXGI_FORMAT_R16G16B16A16_UNORM: desc.format = Format::R16G16B16A16_UNORM; break;
						case dds::DXGI_FORMAT_R16G16B16A16_UINT: desc.format = Format::R16G16B16A16_UINT; break;
						case dds::DXGI_FORMAT_R16G16B16A16_SNORM: desc.format = Format::R16G16B16A16_SNORM; break;
						case dds::DXGI_FORMAT_R16G16B16A16_SINT: desc.format = Format::R16G16B16A16_SINT; break;
						case dds::DXGI_FORMAT_R32G32_FLOAT: desc.format = Format::R32G32_FLOAT; break;
						case dds::DXGI_FORMAT_R32G32_UINT: desc.format = Format::R32G32_UINT; break;
						case dds::DXGI_FORMAT_R32G32_SINT: desc.format = Format::R32G32_SINT; break;
						case dds::DXGI_FORMAT_R10G10B10A2_UNORM: desc.format = Format::R10G10B10A2_UNORM; break;
						case dds::DXGI_FORMAT_R10G10B10A2_UINT: desc.format = Format::R10G10B10A2_UINT; break;
						case dds::DXGI_FORMAT_R11G11B10_FLOAT: desc.format = Format::R11G11B10_FLOAT; break;
						case dds::DXGI_FORMAT_R9G9B9E5_SHAREDEXP: desc.format = Format::R9G9B9E5_SHAREDEXP; break;
						case dds::DXGI_FORMAT_B8G8R8X8_UNORM: desc.format = Format::B8G8R8A8_UNORM; break;
						case dds::DXGI_FORMAT_B8G8R8A8_UNORM: desc.format = Format::B8G8R8A8_UNORM; break;
						case dds::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: desc.format = Format::B8G8R8A8_UNORM_SRGB; break;
						case dds::DXGI_FORMAT_R8G8B8A8_UNORM: desc.format = Format::R8G8B8A8_UNORM; break;
						case dds::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: desc.format = Format::R8G8B8A8_UNORM_SRGB; break;
						case dds::DXGI_FORMAT_R8G8B8A8_UINT: desc.format = Format::R8G8B8A8_UINT; break;
						case dds::DXGI_FORMAT_R8G8B8A8_SNORM: desc.format = Format::R8G8B8A8_SNORM; break;
						case dds::DXGI_FORMAT_R8G8B8A8_SINT: desc.format = Format::R8G8B8A8_SINT; break;
						case dds::DXGI_FORMAT_R16G16_FLOAT: desc.format = Format::R16G16_FLOAT; break;
						case dds::DXGI_FORMAT_R16G16_UNORM: desc.format = Format::R16G16_UNORM; break;
						case dds::DXGI_FORMAT_R16G16_UINT: desc.format = Format::R16G16_UINT; break;
						case dds::DXGI_FORMAT_R16G16_SNORM: desc.format = Format::R16G16_SNORM; break;
						case dds::DXGI_FORMAT_R16G16_SINT: desc.format = Format::R16G16_SINT; break;
						case dds::DXGI_FORMAT_D32_FLOAT: desc.format = Format::D32_FLOAT; break;
						case dds::DXGI_FORMAT_R32_FLOAT: desc.format = Format::R32_FLOAT; break;
						case dds::DXGI_FORMAT_R32_UINT: desc.format = Format::R32_UINT; break;
						case dds::DXGI_FORMAT_R32_SINT: desc.format = Format::R32_SINT; break;
						case dds::DXGI_FORMAT_R8G8_UNORM: desc.format = Format::R8G8_UNORM; break;
						case dds::DXGI_FORMAT_R8G8_UINT: desc.format = Format::R8G8_UINT; break;
						case dds::DXGI_FORMAT_R8G8_SNORM: desc.format = Format::R8G8_SNORM; break;
						case dds::DXGI_FORMAT_R8G8_SINT: desc.format = Format::R8G8_SINT; break;
						case dds::DXGI_FORMAT_R16_FLOAT: desc.format = Format::R16_FLOAT; break;
						case dds::DXGI_FORMAT_D16_UNORM: desc.format = Format::D16_UNORM; break;
						case dds::DXGI_FORMAT_R16_UNORM: desc.format = Format::R16_UNORM; break;
						case dds::DXGI_FORMAT_R16_UINT: desc.format = Format::R16_UINT; break;
						case dds::DXGI_FORMAT_R16_SNORM: desc.format = Format::R16_SNORM; break;
						case dds::DXGI_FORMAT_R16_SINT: desc.format = Format::R16_SINT; break;
						case dds::DXGI_FORMAT_R8_UNORM: desc.format = Format::R8_UNORM; break;
						case dds::DXGI_FORMAT_R8_UINT: desc.format = Format::R8_UINT; break;
						case dds::DXGI_FORMAT_R8_SNORM: desc.format = Format::R8_SNORM; break;
						case dds::DXGI_FORMAT_R8_SINT: desc.format = Format::R8_SINT; break;
						case dds::DXGI_FORMAT_BC1_UNORM: desc.format = Format::BC1_UNORM; break;
						case dds::DXGI_FORMAT_BC1_UNORM_SRGB: desc.format = Format::BC1_UNORM_SRGB; break;
						case dds::DXGI_FORMAT_BC2_UNORM: desc.format = Format::BC2_UNORM; break;
						case dds::DXGI_FORMAT_BC2_UNORM_SRGB: desc.format = Format::BC2_UNORM_SRGB; break;
						case dds::DXGI_FORMAT_BC3_UNORM: desc.format = Format::BC3_UNORM; break;
						case dds::DXGI_FORMAT_BC3_UNORM_SRGB: desc.format = Format::BC3_UNORM_SRGB; break;
						case dds::DXGI_FORMAT_BC4_UNORM: desc.format = Format::BC4_UNORM; break;
						case dds::DXGI_FORMAT_BC4_SNORM: desc.format = Format::BC4_SNORM; break;
						case dds::DXGI_FORMAT_BC5_UNORM: desc.format = Format::BC5_UNORM; break;
						case dds::DXGI_FORMAT_BC5_SNORM: desc.format = Format::BC5_SNORM; break;
						case dds::DXGI_FORMAT_BC6H_SF16: desc.format = Format::BC6H_SF16; break;
						case dds::DXGI_FORMAT_BC6H_UF16: desc.format = Format::BC6H_UF16; break;
						case dds::DXGI_FORMAT_BC7_UNORM: desc.format = Format::BC7_UNORM; break;
						case dds::DXGI_FORMAT_BC7_UNORM_SRGB: desc.format = Format::BC7_UNORM_SRGB; break;
						default:
							assert(0); // incoming format is not supported 
							break;
						}

						if (desc.format == Format::BC4_UNORM || desc.format == Format::BC4_SNORM)
						{
							desc.swizzle.r = ComponentSwizzle::R;
							desc.swizzle.g = ComponentSwizzle::R;
							desc.swizzle.b = ComponentSwizzle::R;
							desc.swizzle.a = ComponentSwizzle::ONE;
						}
						if (desc.format == Format::BC5_UNORM || desc.format == Format::BC5_SNORM)
						{
							desc.swizzle.r = ComponentSwizzle::R;
							desc.swizzle.g = ComponentSwizzle::G;
							desc.swizzle.b = ComponentSwizzle::ONE;
							desc.swizzle.a = ComponentSwizzle::ONE;
						}

						if (header.is_1d())
						{
							desc.type = TextureDesc::Type::TEXTURE_1D;
						}
						else if (header.is_3d())
						{
							desc.type = TextureDesc::Type::TEXTURE_3D;
						}

						if (IsFormatBlockCompressed(desc.format))
						{
							desc.width = AlignTo(desc.width, GetFormatBlockSize(desc.format));
							desc.height = AlignTo(desc.height, GetFormatBlockSize(desc.format));
						}

						wi::vector<SubresourceData> initdata_heap;
						SubresourceData initdata_stack[16] = {};
						SubresourceData* initdata = nullptr;

						// Determine if we need heap allocation for initdata, or it is small enough for stack:
						if (desc.array_size * desc.mip_levels < arraysize(initdata_stack))
						{
							initdata = initdata_stack;
						}
						else
						{
							initdata_heap.resize(desc.array_size * desc.mip_levels);
							initdata = initdata_heap.data();
						}

						uint32_t subresource_index = 0;
						for (uint32_t slice = 0; slice < desc.array_size; ++slice)
						{
							for (uint32_t mip = 0; mip < desc.mip_levels; ++mip)
							{
								SubresourceData& subresourceData = initdata[subresource_index++];
								subresourceData.data_ptr = filedata + header.mip_offset(mip, slice);
								subresourceData.row_pitch = header.row_pitch(mip);
								subresourceData.slice_pitch = header.slice_pitch(mip);
							}
						}

						int mip_offset = 0;
						if (has_flag(flags, Flags::STREAMING))
						{
							// Remember full mipcount for streaming:
							resource->streaming_texture.mip_count = desc.mip_levels;
							// For streaming, remember relative memory offsets for mip levels:
							for (uint32_t slice = 0; slice < desc.array_size; ++slice)
							{
								for (uint32_t mip = 0; mip < desc.mip_levels; ++mip)
								{
									auto& streaming_data = resource->streaming_texture.streaming_data[mip];
									streaming_data.data_offset = header.mip_offset(mip, slice);
									streaming_data.row_pitch = header.row_pitch(mip);
									streaming_data.slice_pitch = header.slice_pitch(mip);
								}
							}
							// Reduce mip map count that will be uploaded to GPU:
							//
							// GGMAX 1.73: the halving must respect block-compression alignment.
							// A BC resource's TOP mip must be a multiple of the 4x4 block size
							// (sub-mips are exempt, which is why the full-size load is fine). Halving
							// a legal size can produce an illegal one -- 500 is a multiple of 4, 250
							// is not -- and the reduced desc is then an invalid BC resource.
							// D3D12's GetCopyableFootprints REJECTS it and writes 0xFFFF.. sentinels
							// into every output, after which the upload loop memcpy'd 4-billion rows
							// off the end of the file buffer. That was the load crash on "Trapped"
							// (DOOR1_surface.dds, 500x500 DXT1) and "RPG Template".
							// Stopping early only means such a texture keeps a larger base mip.
							const uint32_t format_block_size = GetFormatBlockSize(desc.format);
							while (desc.mip_levels > 1 && desc.depth == 1 && desc.array_size == 1 && ComputeTextureMemorySizeInBytes(desc) > streaming_texture_min_size)
							{
								const uint32_t next_width = desc.width >> 1;
								const uint32_t next_height = desc.height >> 1;
								if (format_block_size > 1
									&& ((next_width % format_block_size) != 0 || (next_height % format_block_size) != 0))
								{
									break; // next step would be an illegal top-level size for this format
								}
								desc.width = next_width;
								desc.height = next_height;
								desc.mip_levels -= 1;
								mip_offset++;
							}
							resource->streaming_texture.min_lod_clamp_absolute = (float)mip_offset;
						}

						// GGMAX 1.73 DIAG: breadcrumb EVERY DDS upload (streaming or not), flushed
						// per line, so a fault inside CreateTexture's memcpy names the exact texture.
						// The last line in stream_load.txt when the process dies IS the offender.
						// It must cover non-streaming loads too, otherwise "the last streaming
						// texture" gets blamed for a fault that happened on the next plain one.
						if (gg_stream_load_trace)
						{
							size_t last_byte_needed = 0;
							for (uint32_t m = 0; m < desc.mip_levels; ++m)
							{
								const size_t off = (size_t)header.mip_offset(mip_offset + m, 0);
								last_byte_needed = std::max(last_byte_needed, off + header.slice_pitch(mip_offset + m));
							}
							const std::string trace_path = wi::helper::GetDirectoryFromPath(wi::helper::GetExecutablePath()) + "stream_load.txt";
							std::ofstream trace(trace_path, std::ios::app);
							if (trace)
							{
								trace << (has_flag(flags, Flags::STREAMING) ? "STREAM " : "plain  ")
									<< name
									<< " | file " << header.width() << "x" << header.height()
									<< " mips " << header.mip_levels()
									<< " arr " << header.array_size()
									<< " cube " << (header.is_cubemap() ? 1 : 0)
									<< " fmt " << (uint32_t)desc.format
									<< " | upload " << desc.width << "x" << desc.height
									<< " mips " << desc.mip_levels
									<< " mip_offset " << mip_offset
									<< " | filesize " << filesize
									<< " needs " << last_byte_needed
									<< (last_byte_needed > filesize ? "   *** OVERRUN ***" : "")
									<< std::endl; // endl flushes: we may die on the next statement
							}
						}

						success = device->CreateTexture(&desc, initdata + mip_offset, &resource->texture);
						device->SetName(&resource->texture, name.c_str());

						Format srgb_format = GetFormatSRGB(desc.format);
						if (srgb_format != Format::UNKNOWN && srgb_format != desc.format)
						{
							resource->srgb_subresource = device->CreateSubresource(
								&resource->texture,
								SubresourceType::SRV,
								0, -1,
								0, -1,
								&srgb_format
							);
						}
					}
					else assert(0); // failed to load DDS

				}
				else if (!ext.compare("HDR"))
				{
					flags &= ~Flags::STREAMING; // disable streaming
					int height, width, channels; // stb_image
					float* data = stbi_loadf_from_memory(filedata, (int)filesize, &width, &height, &channels, 0);
					static constexpr bool allow_packing = true; // we now always assume that we won't need full precision float textures, so pack them for memory saving

					if (data != nullptr)
					{
						TextureDesc desc;
						desc.width = (uint32_t)width;
						desc.height = (uint32_t)height;
						switch (channels)
						{
						default:
						case 4:
							if (allow_packing)
							{
								desc.format = Format::R16G16B16A16_FLOAT;
								const XMFLOAT4* data_full = (const XMFLOAT4*)data;
								XMHALF4* data_packed = (XMHALF4*)data;
								for (int i = 0; i < width * height; ++i)
								{
									XMStoreHalf4(data_packed + i, XMLoadFloat4(data_full + i));
								}
							}
							else
							{
								desc.format = Format::R32G32B32A32_FLOAT;
							}
							break;
						case 3:
							if (allow_packing)
							{
								desc.format = Format::R9G9B9E5_SHAREDEXP;
								const XMFLOAT3* data_full = (const XMFLOAT3*)data;
								XMFLOAT3SE* data_packed = (XMFLOAT3SE*)data;
								for (int i = 0; i < width * height; ++i)
								{
									XMStoreFloat3SE(data_packed + i, XMLoadFloat3(data_full + i));
								}
							}
							else
							{
								desc.format = Format::R32G32B32_FLOAT;
							}
							break;
						case 2:
							if (allow_packing)
							{
								desc.format = Format::R16G16_FLOAT;
								const XMFLOAT2* data_full = (const XMFLOAT2*)data;
								XMHALF2* data_packed = (XMHALF2*)data;
								for (int i = 0; i < width * height; ++i)
								{
									XMStoreHalf2(data_packed + i, XMLoadFloat2(data_full + i));
								}
							}
							else
							{
								desc.format = Format::R32G32_FLOAT;
							}
							break;
						case 1:
							if (allow_packing)
							{
								desc.format = Format::R16_FLOAT;
								HALF* data_packed = (HALF*)data;
								for (int i = 0; i < width * height; ++i)
								{
									data_packed[i] = XMConvertFloatToHalf(data[i]);
								}
							}
							else
							{
								desc.format = Format::R32_FLOAT;
							}
							break;
						}
						desc.bind_flags = BindFlag::SHADER_RESOURCE;
						desc.mip_levels = 1;
						SubresourceData InitData;
						InitData.data_ptr = data;
						InitData.row_pitch = width * GetFormatStride(desc.format);
						success = device->CreateTexture(&desc, &InitData, &resource->texture);
						device->SetName(&resource->texture, name.c_str());

						stbi_image_free(data);
					}
				}
				else
				{
					// png, tga, jpg, etc. loader:
					flags &= ~Flags::STREAMING; // disable streaming
					int height = 0, width = 0, channels = 0;
					bool is_16bit = false;
					Format format = Format::R8G8B8A8_UNORM;
					Format bc_format = Format::BC3_UNORM;
					Swizzle swizzle = { ComponentSwizzle::R, ComponentSwizzle::G, ComponentSwizzle::B, ComponentSwizzle::A };

					void* rgba;
					if (!has_flag(flags, Flags::IMPORT_COLORGRADINGLUT) && stbi_is_16_bit_from_memory(filedata, (int)filesize))
					{
						is_16bit = true;
						rgba = stbi_load_16_from_memory(filedata, (int)filesize, &width, &height, &channels, 0);
						switch (channels)
						{
						case 1:
							format = Format::R16_UNORM;
							bc_format = Format::BC4_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::ONE };
							break;
						case 2:
							format = Format::R16G16_UNORM;
							bc_format = Format::BC5_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::G };
							break;
						case 3:
						{
							// Graphics API doesn't support 3 channel formats, so need to expand to RGBA:
							struct Color3
							{
								uint16_t r, g, b;
							};
							const Color3* color3 = (const Color3*)rgba;
							wi::Color16* color4 = (wi::Color16*)malloc(width * height * sizeof(wi::Color16));
							for (int i = 0; i < width * height; ++i)
							{
								color4[i].setR(color3[i].r);
								color4[i].setG(color3[i].g);
								color4[i].setB(color3[i].b);
								color4[i].setA(65535);
							}
							free(rgba);
							rgba = color4;
							format = Format::R16G16B16A16_UNORM;
							bc_format = Format::BC1_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::G, ComponentSwizzle::B, ComponentSwizzle::ONE };
						}
						break;
						case 4:
						default:
							format = Format::R16G16B16A16_UNORM;
							bc_format = Format::BC3_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::G, ComponentSwizzle::B, ComponentSwizzle::A };
							break;
						}
					}
					else
					{
						rgba = stbi_load_from_memory(filedata, (int)filesize, &width, &height, &channels, 0);
						switch (channels)
						{
						case 1:
							format = Format::R8_UNORM;
							bc_format = Format::BC4_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::ONE };
							break;
						case 2:
							format = Format::R8G8_UNORM;
							bc_format = Format::BC5_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::G };
							break;
						case 3:
						{
							// Graphics API doesn't support 3 channel formats, so need to expand to RGBA:
							struct Color3
							{
								uint8_t r, g, b;
							};
							const Color3* color3 = (const Color3*)rgba;
							wi::Color* color4 = (wi::Color*)malloc(width * height * sizeof(wi::Color));
							for (int i = 0; i < width * height; ++i)
							{
								color4[i].setR(color3[i].r);
								color4[i].setG(color3[i].g);
								color4[i].setB(color3[i].b);
								color4[i].setA(255);
							}
							free(rgba);
							rgba = color4;
							format = Format::R8G8B8A8_UNORM;
							bc_format = Format::BC1_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::G, ComponentSwizzle::B, ComponentSwizzle::ONE };
						}
						break;
						case 4:
						default:
							format = Format::R8G8B8A8_UNORM;
							bc_format = Format::BC3_UNORM;
							swizzle = { ComponentSwizzle::R, ComponentSwizzle::G, ComponentSwizzle::B, ComponentSwizzle::A };
							break;
						}
					}

					if (rgba != nullptr)
					{
						TextureDesc desc;
						desc.height = uint32_t(height);
						desc.width = uint32_t(width);
						desc.layout = ResourceState::SHADER_RESOURCE;
						desc.format = format;
						desc.swizzle = swizzle;

						if (has_flag(flags, Flags::IMPORT_COLORGRADINGLUT))
						{
							if (desc.type != TextureDesc::Type::TEXTURE_2D ||
								desc.width != 256 ||
								desc.height != 16 ||
								format != Format::R8G8B8A8_UNORM)
							{
								wi::helper::messageBox("The Dimensions must be 256 x 16 for color grading LUT and format must be RGB or RGBA!", "Error");
							}
							else
							{
								uint32_t data[16 * 16 * 16];
								int pixel = 0;
								for (int z = 0; z < 16; ++z)
								{
									for (int y = 0; y < 16; ++y)
									{
										for (int x = 0; x < 16; ++x)
										{
											int coord = x + y * 256 + z * 16;
											data[pixel++] = ((uint32_t*)rgba)[coord];
										}
									}
								}

								desc.type = TextureDesc::Type::TEXTURE_3D;
								desc.width = 16;
								desc.height = 16;
								desc.depth = 16;
								desc.bind_flags = BindFlag::SHADER_RESOURCE;
								SubresourceData InitData;
								InitData.data_ptr = data;
								InitData.row_pitch = 16 * sizeof(uint32_t);
								InitData.slice_pitch = 16 * InitData.row_pitch;
								success = device->CreateTexture(&desc, &InitData, &resource->texture);
								device->SetName(&resource->texture, name.c_str());
							}
						}
						else
						{
							desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
							desc.mip_levels = GetMipCount(desc.width, desc.height);
							desc.usage = Usage::DEFAULT;
							desc.layout = ResourceState::SHADER_RESOURCE;
							desc.misc_flags = ResourceMiscFlag::TYPED_FORMAT_CASTING;

							uint32_t mipwidth = width;
							SubresourceData init_data[16];
							for (uint32_t mip = 0; mip < desc.mip_levels; ++mip)
							{
								init_data[mip].data_ptr = rgba; // attention! we don't fill the mips here correctly, just always point to the mip0 data by default. Mip levels will be created using compute shader when needed!
								init_data[mip].row_pitch = uint32_t(mipwidth * GetFormatStride(desc.format));
								mipwidth = std::max(1u, mipwidth / 2);
							}

							success = device->CreateTexture(&desc, init_data, &resource->texture);
							device->SetName(&resource->texture, name.c_str());
							device->CreateMipgenSubresources(resource->texture);

							// This part must be AFTER mip level subresource creation:
							Format srgb_format = GetFormatSRGB(desc.format);
							if (srgb_format != Format::UNKNOWN && srgb_format != desc.format)
							{
								resource->srgb_subresource = device->CreateSubresource(
									&resource->texture,
									SubresourceType::SRV,
									0, -1,
									0, -1,
									&srgb_format
								);
							}

							wi::renderer::AddDeferredMIPGen(resource->texture, true);

							if (has_flag(flags, Flags::IMPORT_BLOCK_COMPRESSED))
							{
								// Schedule additional task to compress into BC format and replace resource texture:
								Texture uncompressed_src = std::move(resource->texture);
								resource->srgb_subresource = -1;

								desc.format = bc_format;

								if (has_flag(flags, Flags::IMPORT_NORMALMAP))
								{
									desc.format = Format::BC5_UNORM;
									desc.swizzle = { ComponentSwizzle::R, ComponentSwizzle::G, ComponentSwizzle::ONE, ComponentSwizzle::ONE };
								}

								desc.bind_flags = BindFlag::SHADER_RESOURCE;

								const uint32_t block_size = GetFormatBlockSize(desc.format);
								desc.width = align(desc.width, block_size);
								desc.height = align(desc.height, block_size);
								desc.mip_levels = GetMipCount(desc.width, desc.height, 1, block_size);

								success = device->CreateTexture(&desc, nullptr, &resource->texture);
								device->SetName(&resource->texture, name.c_str());

								// This part must be AFTER mip level subresource creation:
								Format srgb_format = GetFormatSRGB(desc.format);
								if (srgb_format != Format::UNKNOWN && srgb_format != desc.format)
								{
									resource->srgb_subresource = device->CreateSubresource(
										&resource->texture,
										SubresourceType::SRV,
										0, -1,
										0, -1,
										&srgb_format
									);
								}

								wi::renderer::AddDeferredBlockCompression(uncompressed_src, resource->texture);
							}
						}
					}
					stbi_image_free(rgba);
				}
			}
			break;

			case DataType::SOUND:
			{
				success = wi::audio::CreateSound(filedata, filesize, &resource->sound);
			}
			break;

			case DataType::SCRIPT:
			{
				resource->script.resize(filesize);
				std::memcpy(resource->script.data(), filedata, filesize);
				resource->script_hash = wi::helper::string_hash(resource->script.c_str());
				success = true;
			}
			break;

			case DataType::VIDEO_MP4:
			{
				success = wi::video::CreateVideoMP4(filedata, filesize, &resource->video);
			}
			break;

			case DataType::VIDEO_H264_RAW:
			{
				success = wi::video::CreateVideoH264RAW(filedata, filesize, &resource->video);
			}
			break;

			case DataType::FONTSTYLE:
			{
				resource->font_style = wi::font::AddFontStyle(name, filedata, filesize, true);
				success = resource->font_style >= 0;
			}
			break;

			};

			if (!resource->filedata.empty() && !has_flag(flags, Flags::IMPORT_RETAIN_FILEDATA) && !has_flag(flags, Flags::IMPORT_DELAY))
			{
				// file data can be discarded:
				resource->filedata.clear();
				resource->filedata.shrink_to_fit();
			}

			return success;
		}

		Resource Load(
			const std::string& name,
			Flags flags,
			const uint8_t* filedata,
			size_t filesize,
			const std::string& container_filename,
			size_t container_fileoffset
		)
		{
			locker.lock();
			wi::allocator::weak_ptr<ResourceInternal>& weak_resource = resources[name];
			wi::allocator::shared_ptr<ResourceInternal> resource = weak_resource.lock();

			// GGMAX 1.52b tripwire: a cache hit whose internal names a DIFFERENT file means the
			// name->resource binding was hijacked (the pooled weak_ptr underflow fixed by 1.52,
			// or any undiscovered sibling). Self-heal by treating it as a cache miss, and log
			// the evidence next to the EXE. Post-1.52 this should never fire.
			if (resource != nullptr && !resource->filename.empty() && resource->filename != name)
			{
				const std::string report_path = wi::helper::GetDirectoryFromPath(wi::helper::GetExecutablePath()) + "resource_hijack.txt";
				std::ofstream hijack_report(report_path, std::ios::app);
				if (hijack_report)
				{
					hijack_report << "requested \"" << name << "\" but cache entry holds \"" << resource->filename << "\" — rebinding fresh\n";
				}
				resource.reset();
			}

			uint64_t timestamp = 0;
			if(!container_filename.empty())
			{
				timestamp = wi::helper::FileTimestamp(container_filename);
			}
			else
			{
				timestamp = wi::helper::FileTimestamp(name);
			}

			if (resource == nullptr || resource->timestamp < timestamp)
			{
				resource = wi::allocator::make_shared<ResourceInternal>();
				resources[name] = resource;
				resource->filename = name;

				// Rememeber the streaming file parameters, which is either the resource filename,
				//	or it can be a specific filename and offset in the case when the file contained multiple resources
				if (container_filename.empty())
				{
					resource->container_filename = name;
				}
				else
				{
					resource->container_filename = container_filename;
				}
				resource->container_filesize = filesize;
				resource->container_fileoffset = container_fileoffset;

				if (filedata != nullptr && resource->filedata.empty() && (has_flag(flags, Flags::IMPORT_RETAIN_FILEDATA) || has_flag(flags, Flags::IMPORT_DELAY)))
				{
					// resource was loaded with external filedata, and we want to retain filedata
					//	this must also happen when using IMPORT_DELAY!
					resource->filedata.resize(filesize);
					std::memcpy(resource->filedata.data(), filedata, filesize);
				}
			}
			else
			{
				if (!has_flag(flags, Flags::IMPORT_DELAY) && has_flag(resource->flags, Flags::IMPORT_DELAY))
				{
					// If this is not an IMPORT_DELAY load, but this resource load was incomplete, using IMPORT_DELAY,
					//	then continue loading it as normal from existing file data and remove IMPORT_DELAY flag from it
					resource->flags &= ~Flags::IMPORT_DELAY;
				}
				else
				{
					resource_log("\tResource reused: %s", name.c_str());
					Resource retVal;
					retVal.internal_state = resource;
					locker.unlock();
					return retVal;
				}
			}
			locker.unlock();

			if (filedata == nullptr || filesize == 0)
			{
				if (resource->filedata.empty())
				{
					if (!wi::helper::FileRead(resource->container_filename, resource->filedata, resource->container_filesize, resource->container_fileoffset))
					{
						resource.reset();
						return Resource();
					}
				}
				filedata = resource->filedata.data();
				filesize = resource->filedata.size();
			}

			flags |= resource->flags;

			bool success = false;

			if (has_flag(flags, Flags::IMPORT_DELAY))
			{
				success = true;
			}
			else
			{
				resource_log("\tResource loading: %s", name.c_str());
				success = LoadResourceDirectly(name, flags, filedata, filesize, resource.get());
			}

			if (success)
			{
				resource->flags = flags;
				resource->timestamp = timestamp;

				Resource retVal;
				retVal.internal_state = resource;
				return retVal;
			}

			return Resource();
		}

		bool Contains(const std::string& name)
		{
			bool result = false;
			locker.lock();
			auto it = resources.find(name);
			if (it != resources.end())
			{
				auto resource = it->second.lock();
				result = resource != nullptr;
			}
			locker.unlock();
			return result;
		}

		void Clear()
		{
			locker.lock();
			resources.clear();
			locker.unlock();
		}

		wi::jobsystem::context streaming_ctx;
		wi::vector<wi::allocator::shared_ptr<ResourceInternal>> streaming_texture_jobs;
		struct StreamingTextureReplace
		{
			wi::allocator::shared_ptr<ResourceInternal> resource;
			Texture texture;
			int srgb_subresource = -1;
		};
		std::mutex streaming_replacement_mutex;
		wi::vector<StreamingTextureReplace> streaming_texture_replacements;
		float streaming_threshold = 0.8f;
		float streaming_fade_speed = 4;

		void SetStreamingMemoryThreshold(float value)
		{
			std::scoped_lock lck(locker);
			streaming_threshold = value;
		}

		float GetStreamingMemoryThreshold()
		{
			std::scoped_lock lck(locker);
			return streaming_threshold;
		}

		// GGMAX DIAG (2026-07-26 reload-corruption hunt): pause the whole texture-streaming
		// system (no min-lod updates, no mip stream in/out, no texture replacements).
		// Textures stay at their initially-loaded resolution. A/B probe: if the reload
		// corruption (wrong texture content) disappears with streaming paused, the
		// streaming path is the writer. Driven by harness SET_STREAMING <0|1>.
		// CONFIRMED 2026-07-26: streaming paused = zero corruption across reloads.
		bool gg_streaming_paused = false;

		// GGMAX 1.69: texture-streaming observability (harness GET_PERF_DATA STREAM line).
		// enrolled = resources gathered into the last streaming job pass; replaced = texture
		// replacements applied on the main thread since launch; resident/full = VRAM bytes of
		// enrolled textures now vs their complete mip chains; mem_permille = GPU usage/budget.
		std::atomic<uint32_t> gg_dbg_stream_enrolled{ 0 };
		std::atomic<uint32_t> gg_dbg_stream_replaced{ 0 };
		std::atomic<unsigned long long> gg_dbg_stream_resident_bytes{ 0 };
		std::atomic<unsigned long long> gg_dbg_stream_full_bytes{ 0 };
		std::atomic<uint32_t> gg_dbg_stream_mem_permille{ 0 };
		// GGMAX 1.69: streaming-job liveness probes (job launched / job completed / skipped-because-busy)
		std::atomic<unsigned long long> gg_dbg_stream_job_starts{ 0 };
		std::atomic<unsigned long long> gg_dbg_stream_job_ends{ 0 };
		std::atomic<unsigned long long> gg_dbg_stream_busy_skips{ 0 };
		// GGMAX 1.69: per-pass decision census (stored at job end) + max requested resolution seen
		std::atomic<uint32_t> gg_dbg_stream_dec_req0{ 0 };      // requested == 0 (no feedback for this resource)
		std::atomic<uint32_t> gg_dbg_stream_dec_reqlow{ 0 };    // requested below current size (wants smaller/equal)
		std::atomic<uint32_t> gg_dbg_stream_dec_nomips{ 0 };    // wants IN but no more mips in container
		std::atomic<uint32_t> gg_dbg_stream_dec_cancel{ 0 };    // wants IN but +1 mip would overshoot
		std::atomic<uint32_t> gg_dbg_stream_dec_in{ 0 };        // streamed IN
		std::atomic<uint32_t> gg_dbg_stream_dec_out{ 0 };       // streamed OUT
		std::atomic<uint32_t> gg_dbg_stream_max_req{ 0 };       // largest requested resolution seen (all-time)

		// GGMAX 1.73 DIAG: per-load breadcrumb trace of streaming DDS uploads (stream_load.txt).
		// Off by default — it writes a flushed line per enrolled texture, which is far too much
		// I/O for normal running. Harness SET_TEXSTREAMTRACE 1 arms it before a suspect load.
		bool gg_stream_load_trace = false;

		// GGMAX 1.73: streaming bounds-guard reporting. Every rejection the streaming job makes
		// is logged once per (resource, reason) to stream_guard.txt next to the EXE, with the
		// numbers needed to tell WHICH invariant broke. On a healthy build this file never
		// appears — same tripwire discipline as alloc_tripwire.txt / resource_hijack.txt.
		std::atomic<uint32_t> gg_dbg_stream_guard_rejects{ 0 };
		static std::mutex gg_stream_guard_mutex;
		static std::set<std::string> gg_stream_guard_seen;
		void gg_stream_guard_report(
			const std::string& name,
			const std::string& container,
			const char* reason,
			int mip_offset,
			uint32_t mip_levels,
			uint32_t mip_count,
			size_t container_filesize,
			size_t mip_data_offset,
			size_t reserved,
			size_t required_bytes = 0,
			size_t available_bytes = 0
		)
		{
			gg_dbg_stream_guard_rejects.fetch_add(1, std::memory_order_relaxed);

			std::scoped_lock lock(gg_stream_guard_mutex);
			const std::string key = name + "|" + reason;
			if (!gg_stream_guard_seen.insert(key).second)
				return; // already reported this resource for this reason

			// The on-disk size is the whole point of the SHORT READ case: it is what the loader's
			// recorded container_filesize is being compared against.
			size_t ondisk_size = 0;
			{
				std::ifstream probe(container, std::ios::binary | std::ios::ate);
				if (probe.is_open())
					ondisk_size = (size_t)probe.tellg();
			}

			const std::string report_path = wi::helper::GetDirectoryFromPath(wi::helper::GetExecutablePath()) + "stream_guard.txt";
			std::ofstream report(report_path, std::ios::app);
			if (!report)
				return;
			report << "REJECT [" << reason << "]\n";
			report << "  resource        : " << name << "\n";
			report << "  container       : " << container << "\n";
			report << "  mip_offset      : " << mip_offset << "  mip_levels " << mip_levels << "  mip_count " << mip_count << "\n";
			report << "  container_size  : " << container_filesize << "   on-disk now " << ondisk_size;
			if (ondisk_size != 0 && container_filesize != 0 && ondisk_size != container_filesize)
				report << "   *** DISK SIZE DIFFERS BY " << (long long)ondisk_size - (long long)container_filesize << " ***";
			report << "\n";
			report << "  mip_data_offset : " << mip_data_offset << "\n";
			if (required_bytes != 0 || available_bytes != 0)
			{
				report << "  required bytes  : " << required_bytes << "\n";
				report << "  available bytes : " << available_bytes;
				if (required_bytes > available_bytes)
					report << "   *** SHORT BY " << (required_bytes - available_bytes) << " ***";
				report << "\n";
			}
			report << "\n";
		}

		// GGMAX 1.44: see wiResourceManager.h. Implemented below UpdateStreamingResources
		// (needs streaming_ctx / replacement queue visibility).

		void UpdateStreamingResources(float dt)
		{
			if (gg_streaming_paused)
				return;
			// If any streaming replacement requests arrived, replace the resources here (main thread):
			streaming_replacement_mutex.lock(); // streaming_replacement_mutex is not a long lock, it can only be held by the single streaming thread, so we don't need to try_lock
			if (!streaming_texture_replacements.empty())
			{
				gg_streaming_descriptor_epoch.fetch_add(1, std::memory_order_relaxed); // GGMAX 1.41: descriptors changed
				gg_dbg_stream_replaced.fetch_add((uint32_t)streaming_texture_replacements.size(), std::memory_order_relaxed); // GGMAX 1.69
			}
			for (auto& replace : streaming_texture_replacements)
			{
				replace.resource->texture = replace.texture;
				replace.resource->srgb_subresource = replace.srgb_subresource;
			}
			streaming_texture_replacements.clear();
			streaming_replacement_mutex.unlock();

			// Update resource min lod clamps smoothly:
			GraphicsDevice* device = GetDevice();
			if (!locker.try_lock()) // Use try lock as this is on the main thread which shouldn't hitch on long locking!
				return; // Streaming is not that important, we can abandon it if some resource loading is holding the lock
			bool gg_any_subresource_recreated = false; // GGMAX 1.41
			for (auto& x : resources)
			{
				wi::allocator::weak_ptr<ResourceInternal>& weak_resource = x.second;
				wi::allocator::shared_ptr<ResourceInternal> resource = weak_resource.lock();
				if (resource != nullptr && resource->texture.IsValid() && has_flag(resource->flags, Flags::STREAMING))
				{
					const TextureDesc& desc = resource->texture.desc;
					const float mip_offset = float(resource->streaming_texture.mip_count - desc.mip_levels);
					float min_lod_clamp_absolute_next = resource->streaming_texture.min_lod_clamp_absolute - dt * streaming_fade_speed;
					min_lod_clamp_absolute_next = std::max(mip_offset, min_lod_clamp_absolute_next);
					if (wi::math::float_equal(min_lod_clamp_absolute_next, resource->streaming_texture.min_lod_clamp_absolute))
						continue;
					resource->streaming_texture.min_lod_clamp_absolute = min_lod_clamp_absolute_next;
					gg_any_subresource_recreated = true; // GGMAX 1.41: descriptors recreated below

					const float min_lod_clamp_relative = min_lod_clamp_absolute_next - mip_offset;

					device->DeleteSubresources(&resource->texture);

					device->CreateSubresource(
						&resource->texture,
						SubresourceType::SRV,
						0, -1,
						0, -1,
						nullptr,
						nullptr,
						nullptr,
						min_lod_clamp_relative
					);
					resource->srgb_subresource = -1;

					Format srgb_format = GetFormatSRGB(desc.format);
					if (srgb_format != Format::UNKNOWN && srgb_format != desc.format)
					{
						resource->srgb_subresource = device->CreateSubresource(
							&resource->texture,
							SubresourceType::SRV,
							0, -1,
							0, -1,
							&srgb_format,
							nullptr,
							nullptr,
							min_lod_clamp_relative
						);
					}
				}
			}

			if (gg_any_subresource_recreated)
			{
				gg_streaming_descriptor_epoch.fetch_add(1, std::memory_order_relaxed); // GGMAX 1.41
			}

			// If previous streaming jobs were not finished, we cancel this until next frame:
			if (wi::jobsystem::IsBusy(streaming_ctx))
			{
				gg_dbg_stream_busy_skips.fetch_add(1, std::memory_order_relaxed); // GGMAX 1.69
				locker.unlock();
				return;
			}

			streaming_texture_jobs.clear();

			static wi::vector<const std::string*> removals; // string ptr to avoid string copies, or string constructions from char*

			// Gather the streaming jobs, unload lost resources:
			for (auto& x : resources)
			{
				if (x.second.expired())
				{
					removals.push_back(&x.first);
					continue;
				}

				wi::allocator::weak_ptr<ResourceInternal>& weak_resource = x.second;
				wi::allocator::shared_ptr<ResourceInternal> resource = weak_resource.lock();
				if (resource != nullptr && resource->texture.IsValid() && resource->streaming_texture.mip_count > 1)
				{
					streaming_texture_jobs.push_back(resource);
				}
			}

			for (auto& x : removals)
			{
				resource_log("\tResource lost: %s", x->c_str());
				resources.erase(*x);
			}
			removals.clear();
			locker.unlock();

			gg_dbg_stream_enrolled.store((uint32_t)streaming_texture_jobs.size(), std::memory_order_relaxed); // GGMAX 1.69

			if (streaming_texture_jobs.empty())
				return;

			// One low priority thread will be responsible for streaming, to not cause any hitching while rendering:
			streaming_ctx.priority = wi::jobsystem::Priority::Streaming;
			wi::jobsystem::Execute(streaming_ctx, [](wi::jobsystem::JobArgs args) {
				gg_dbg_stream_job_starts.fetch_add(1, std::memory_order_relaxed); // GGMAX 1.69
				unsigned long long gg_resident = 0, gg_full = 0; // GGMAX 1.69
				uint32_t dc_req0 = 0, dc_reqlow = 0, dc_nomips = 0, dc_cancel = 0, dc_in = 0, dc_out = 0; // GGMAX 1.69 decision census
				for(auto& resource : streaming_texture_jobs)
				{
					TextureDesc desc = resource->texture.desc;
					{
						// GGMAX 1.69: byte accounting BEFORE desc is mutated by stream in/out below
						gg_resident += ComputeTextureMemorySizeInBytes(desc);
						TextureDesc full = desc;
						const uint32_t dropped = resource->streaming_texture.mip_count - desc.mip_levels;
						full.width <<= dropped;
						full.height <<= dropped;
						full.mip_levels = resource->streaming_texture.mip_count;
						gg_full += ComputeTextureMemorySizeInBytes(full);
					}
					uint32_t requested_resolution = resource->streaming_resolution.fetch_and(0); // set to zero while returning prev value
					if (requested_resolution > 0)
					{
						requested_resolution = 1u << (31u - firstbithigh(requested_resolution)); // largest power of two
						gg_dbg_stream_max_req.fetch_or(requested_resolution, std::memory_order_relaxed); // GGMAX 1.69: pow2 -> bitmask census of request magnitudes
					}
					GraphicsDevice* device = GetDevice();
					const GraphicsDevice::MemoryUsage memory_usage = device->GetMemoryUsage();
					const float memory_percent = float(double(memory_usage.usage) / double(memory_usage.budget));
					gg_dbg_stream_mem_permille.store((uint32_t)(memory_percent * 1000.0f), std::memory_order_relaxed); // GGMAX 1.69
					const bool memory_shortage = memory_percent > streaming_threshold;
					const bool stream_in = requested_resolution >= std::min(desc.width, desc.height);
					const uint32_t target_unload_delay = memory_shortage ? 4 : 255;

					int mip_offset = int(resource->streaming_texture.mip_count - desc.mip_levels);
					if (stream_in)
					{
						resource->streaming_unload_delay = 0; // unloading will be immediately halted
						if (mip_offset == 0)
						{
							dc_nomips++; // GGMAX 1.69
							continue; // There aren't any more mip levels, cancel
						}
						// Mip level streaming IN:
						desc.width <<= 1;
						desc.height <<= 1;
						if (requested_resolution < std::min(desc.width, desc.height))
						{
							dc_cancel++; // GGMAX 1.69
							continue; // Increased resolution would be too much, cancel
						}
						desc.mip_levels++;
						mip_offset--;
						dc_in++; // GGMAX 1.69
					}
					else
					{
						if (requested_resolution == 0) dc_req0++; else dc_reqlow++; // GGMAX 1.69
						resource->streaming_unload_delay++; // one more frame that this wants to unload
						if (resource->streaming_unload_delay < target_unload_delay)
							continue; // only unload mips if it's been wanting to unload for a couple frames, or there is memory shortage
						if (ComputeTextureMemorySizeInBytes(desc) <= streaming_texture_min_size)
							continue; // Don't reduce the texture below, because of min resource alignment, this would not reduce memory usage further
						// Mip level streaming OUT, fast decay:
						// GGMAX 1.73: same block-alignment rule as the initial load reduction —
						// halving a BC texture's top mip below block alignment (500 -> 250) makes
						// the desc illegal and GetCopyableFootprints returns -1 sentinels. See the
						// load-time reduction in LoadResourceDirectly for the full explanation.
						const uint32_t format_block_size = GetFormatBlockSize(desc.format);
						const int mip_offset_before = mip_offset;
						while (ComputeTextureMemorySizeInBytes(desc) > streaming_texture_min_size && desc.width > requested_resolution && desc.height > requested_resolution)
						{
							const uint32_t next_width = desc.width >> 1;
							const uint32_t next_height = desc.height >> 1;
							if (format_block_size > 1
								&& ((next_width % format_block_size) != 0 || (next_height % format_block_size) != 0))
							{
								break;
							}
							desc.width = next_width;
							desc.height = next_height;
							desc.mip_levels--;
							mip_offset++;
						}
						if (mip_offset == mip_offset_before)
						{
							// GGMAX 1.73: nothing could be shed — e.g. a 500x500 BC texture, whose
							// every smaller mip is block-misaligned. Without this the desc is
							// unchanged and we fall through to build a byte-identical replacement
							// texture on every streaming pass: a full file re-read plus re-upload,
							// forever, for no memory saved.
							continue;
						}
						dc_out++; // GGMAX 1.69
					}
					if (desc.mip_levels <= resource->streaming_texture.mip_count)
					{
						// GGMAX 1.73: mip_offset indexes a fixed 16-entry array. Every value here is
						// derived from a file header and a desc that other code can mutate, so range
						// check it rather than trusting it — reading streaming_data[] out of bounds
						// yields a garbage data_offset and turns the upload below into a wild pointer.
						if (mip_offset < 0
						 || mip_offset >= (int)arraysize(resource->streaming_texture.streaming_data)
						 || (size_t)mip_offset + desc.mip_levels > arraysize(resource->streaming_texture.streaming_data)
						 || (uint32_t)mip_offset + desc.mip_levels > resource->streaming_texture.mip_count)
						{
							gg_stream_guard_report(resource->filename, resource->container_filename,
								"mip_offset out of range", mip_offset, desc.mip_levels,
								resource->streaming_texture.mip_count, 0, 0, 0);
							continue;
						}

						// memory offset of the first mip level in current streaming range:
						const size_t mip_data_offset = resource->streaming_texture.streaming_data[mip_offset].data_offset;
						const uint8_t* firstmipdata = resource->filedata.data();

						// GGMAX 1.73: how many bytes are actually readable from firstmipdata. The
						// stock code never tracked this, so a container file that no longer matches
						// the size recorded at load time (short read, replaced/truncated file) fed
						// out-of-range pointers straight into CreateTexture's memcpy.
						size_t available_bytes = 0;

						static wi::vector<uint8_t> streaming_file; // make this static to not reallocate for each file loading
						if (firstmipdata == nullptr)
						{
							// If file data is not available, then open the file partially with the streaming file parameters:
							if (resource->container_filesize < mip_data_offset)
							{
								gg_stream_guard_report(resource->filename, resource->container_filename,
									"container_filesize < mip_data_offset", mip_offset, desc.mip_levels,
									resource->streaming_texture.mip_count, resource->container_filesize, mip_data_offset, 0);
								continue;
							}
							size_t filesize = resource->container_filesize - mip_data_offset;
							size_t fileoffset = resource->container_fileoffset + mip_data_offset;
							if (!wi::helper::FileRead(
								resource->container_filename,
								streaming_file,
								filesize,
								fileoffset
							))
							{
								gg_stream_guard_report(resource->filename, resource->container_filename,
									"FileRead failed", mip_offset, desc.mip_levels,
									resource->streaming_texture.mip_count, resource->container_filesize, mip_data_offset, 0);
								continue;
							}
							firstmipdata = streaming_file.data();
							available_bytes = streaming_file.size();
						}
						else
						{
							// If file data is available, we can use that for streaming:
							if (resource->filedata.size() <= mip_data_offset)
							{
								gg_stream_guard_report(resource->filename, resource->container_filename,
									"retained filedata shorter than mip offset", mip_offset, desc.mip_levels,
									resource->streaming_texture.mip_count, resource->filedata.size(), mip_data_offset, 0);
								continue;
							}
							firstmipdata += mip_data_offset;
							available_bytes = resource->filedata.size() - mip_data_offset;
						}

						// Convert relative to absolute GPU initialization data
						SubresourceData initdata[16] = {};
						size_t required_bytes = 0;
						bool offsets_sane = true;
						for (uint32_t mip = 0; mip < desc.mip_levels; ++mip)
						{
							auto& streaming_data = resource->streaming_texture.streaming_data[mip_offset + mip];
							// Offsets are absolute within the container; anything before the first mip
							// of this range would index behind the buffer we just read.
							if (streaming_data.data_offset < mip_data_offset)
							{
								offsets_sane = false;
								break;
							}
							const size_t relative_offset = streaming_data.data_offset - mip_data_offset;
							required_bytes = std::max(required_bytes, relative_offset + streaming_data.slice_pitch);
							initdata[mip].data_ptr = firstmipdata + relative_offset;
							initdata[mip].row_pitch = streaming_data.row_pitch;
							initdata[mip].slice_pitch = streaming_data.slice_pitch;
						}

						// GGMAX 1.73: THE guard. CreateTexture memcpys slice_pitch bytes per mip out
						// of this buffer; if the file gave us fewer bytes than the header promised,
						// that read runs off the end of the heap block and faults inside memcpy.
						// Skipping the resource costs one texture staying at its current mip level.
						if (!offsets_sane || required_bytes > available_bytes)
						{
							gg_stream_guard_report(resource->filename, resource->container_filename,
								offsets_sane ? "SHORT READ - required > available" : "mip data_offset before range start",
								mip_offset, desc.mip_levels, resource->streaming_texture.mip_count,
								resource->container_filesize, mip_data_offset, 0,
								required_bytes, available_bytes);
							continue;
						}

						// The replacement struct will store the newly created texture until replacement can be made later:
						StreamingTextureReplace replace;
						replace.resource = resource;
						replace.srgb_subresource = -1;
						bool success = device->CreateTexture(&desc, initdata, &replace.texture);
						assert(success);
						device->SetName(&replace.texture, resource->filename.c_str());

						Format srgb_format = GetFormatSRGB(desc.format);
						if (srgb_format != Format::UNKNOWN && srgb_format != desc.format)
						{
							replace.srgb_subresource = device->CreateSubresource(
								&replace.texture,
								SubresourceType::SRV,
								0, -1,
								0, -1,
								&srgb_format
							);
						}

						streaming_replacement_mutex.lock();
						streaming_texture_replacements.push_back(replace);
						streaming_replacement_mutex.unlock();
					}
				}
				gg_dbg_stream_resident_bytes.store(gg_resident, std::memory_order_relaxed); // GGMAX 1.69
				gg_dbg_stream_full_bytes.store(gg_full, std::memory_order_relaxed);
				gg_dbg_stream_dec_req0.store(dc_req0, std::memory_order_relaxed); // GGMAX 1.69 decision census (per pass)
				gg_dbg_stream_dec_reqlow.store(dc_reqlow, std::memory_order_relaxed);
				gg_dbg_stream_dec_nomips.store(dc_nomips, std::memory_order_relaxed);
				gg_dbg_stream_dec_cancel.store(dc_cancel, std::memory_order_relaxed);
				gg_dbg_stream_dec_in.store(dc_in, std::memory_order_relaxed);
				gg_dbg_stream_dec_out.store(dc_out, std::memory_order_relaxed);
				gg_dbg_stream_job_ends.fetch_add(1, std::memory_order_relaxed); // GGMAX 1.69
			});
		}

		// GGMAX 1.69: dump every live resource's streaming state (name, current vs full mip
		// chain, STREAMING flag, live request value) — the authoritative enrolled-set list.
		// Driven by harness DUMP_STREAM2.
		void GG_DumpStreamingResources(const char* path)
		{
			FILE* f = fopen(path, "w");
			if (f == nullptr)
				return;
			locker.lock();
			int total = 0, enrolled = 0, flagged = 0;
			for (auto& x : resources)
			{
				wi::allocator::shared_ptr<ResourceInternal> res = x.second.lock();
				if (res == nullptr)
					continue;
				total++;
				if (!res->texture.IsValid())
					continue;
				const TextureDesc& d = res->texture.desc;
				const bool en = res->streaming_texture.mip_count > 1;
				const bool fl = has_flag(res->flags, Flags::STREAMING);
				if (en) enrolled++;
				if (fl) flagged++;
				fprintf(f, "%s cur=%ux%u/%u fullmips=%u flagSTREAM=%d minlod=%.1f reqNow=%u \"%s\"\n",
					en ? "ENROLLED" : "static  ",
					d.width, d.height, d.mip_levels,
					res->streaming_texture.mip_count,
					fl ? 1 : 0,
					res->streaming_texture.min_lod_clamp_absolute,
					res->streaming_resolution.load(std::memory_order_relaxed),
					x.first.c_str());
			}
			fprintf(f, "total=%d enrolled=%d flagged=%d\n", total, enrolled, flagged);
			locker.unlock();
			fclose(f);
		}

		// GGMAX 1.44: quiesce streaming across an in-place level reload (see header).
		void GGReloadGuardBegin()
		{
			gg_streaming_paused = true;
			wi::jobsystem::Wait(streaming_ctx); // join the in-flight streaming job
			streaming_replacement_mutex.lock();
			// Pending replacements were computed against the dying session's resources —
			// drop them; survivors will simply re-stream under the new session.
			streaming_texture_replacements.clear();
			streaming_replacement_mutex.unlock();
			streaming_texture_jobs.clear(); // release held shared_ptrs to old resources
		}
		void GGReloadGuardEnd()
		{
			gg_streaming_paused = false;
		}

		bool CheckResourcesOutdated()
		{
			std::scoped_lock lck(locker);

			for (auto& x : resources)
			{
				const std::string& name = x.first;
				auto resourceinternal = x.second.lock();
				if (resourceinternal == nullptr)
					continue;

				uint64_t timestamp = wi::helper::FileTimestamp(resourceinternal->filename);
				if (resourceinternal->timestamp < timestamp)
					return true;
			}
			return false;
		}

		void ReloadOutdatedResources()
		{
			std::scoped_lock lck(locker);

			for (auto& x : resources)
			{
				auto resourceinternal = x.second.lock();
				if (resourceinternal == nullptr)
					continue;

				uint64_t timestamp = wi::helper::FileTimestamp(resourceinternal->filename);
				if (resourceinternal->timestamp < timestamp)
				{
					wi::vector<uint8_t> filedata;
					if (wi::helper::FileRead(resourceinternal->filename, filedata))
					{
						if (resourceinternal->streaming_texture.mip_count > 1)
							wi::jobsystem::Wait(streaming_ctx); // reloading a resource that is potentially streaming needs to wait for current streaming job to end
						if (LoadResourceDirectly(resourceinternal->filename, resourceinternal->flags, filedata.data(), filedata.size(), resourceinternal.get()))
						{
							resourceinternal->timestamp = timestamp;
							resourceinternal->container_filename = resourceinternal->filename;
							resourceinternal->container_fileoffset = 0;
							resourceinternal->container_filesize = ~0ull;
							wi::backlog::post("[resourcemanager] reload success: " + resourceinternal->filename);
						}
						else
						{
							wi::backlog::post("[resourcemanager] reload failure - LoadResourceDirectly returned false: " + resourceinternal->filename, wi::backlog::LogLevel::Error);
						}
					}
					else
					{
						wi::backlog::post("[resourcemanager] reload failure - file data could not be read: " + resourceinternal->filename, wi::backlog::LogLevel::Error);
					}
				}
			}
		}

		void Serialize_READ(wi::Archive& archive, ResourceSerializer& seri)
		{
			assert(archive.IsReadMode());
			wi::jobsystem::Wait(streaming_ctx); // stop streaming at this point

			size_t serializable_count = 0;
			archive >> serializable_count;

			struct TempResource
			{
				std::string name;
				const uint8_t* filedata = nullptr;
				size_t filesize = 0;
			};
			wi::vector<TempResource> temp_resources;
			temp_resources.resize(serializable_count);

			wi::jobsystem::context ctx;
			ctx.priority = wi::jobsystem::Priority::Low;

			for (size_t i = 0; i < serializable_count; ++i)
			{
				auto& resource = temp_resources[i];

				archive >> resource.name;
				uint32_t flags_temp;
				archive >> flags_temp;
				// Note: flags not applied here, but they must be read
				//	We don't apply the flags, because they will be requested later by for example materials
				//	If we would apply flags here, then flags from previous session would be applied, that maybe we no longer want (for example RETAIN_FILEDATA)

				// We don't read the file data from archive into a vector like usual, instead map the vector,
				//  this is much faster and we don't need to retain this data after archive lifetime
				archive.MapVector(resource.filedata, resource.filesize);

				size_t file_offset = archive.GetPos() - resource.filesize;
				
				resource.name = archive.GetSourceDirectory() + resource.name;

				if (Contains(resource.name))
					continue;

				// "Loading" the resource can happen asynchronously to serialization of file data, to improve performance
				wi::jobsystem::Execute(ctx, [i, &temp_resources, &seri, &archive, file_offset](wi::jobsystem::JobArgs args) {
					auto& tmp_resource = temp_resources[i];
					Flags flags = Flags::IMPORT_DELAY;
					if (archive.IsCompressionEnabled())
					{
						// If compressed archive, cannot stream from it, retain file data in memory:
						flags |= Flags::IMPORT_RETAIN_FILEDATA;
					}
					auto res = Load(
						tmp_resource.name,
						flags,
						tmp_resource.filedata,
						tmp_resource.filesize,
						archive.GetSourceFileName(),
						file_offset
					);
					static std::mutex seri_locker;
					seri_locker.lock();
					seri.resources.push_back(res);
					seri_locker.unlock();
				});
			}
			wi::jobsystem::Wait(ctx);
		}
		void Serialize_WRITE(wi::Archive& archive, const wi::unordered_set<std::string>& resource_names)
		{
			assert(!archive.IsReadMode());

			wi::jobsystem::Wait(streaming_ctx); // stop streaming at this point

			locker.lock();
			size_t serializable_count = 0;

			if (mode == Mode::NO_EMBEDDING)
			{
				// Simply not serialize any embedded resources
				serializable_count = 0;
				archive << serializable_count;
			}
			else
			{
				// Count embedded resources:
				for (auto& name : resource_names)
				{
					auto it = resources.find(name);
					if (it == resources.end())
						continue;
					wi::allocator::shared_ptr<ResourceInternal> resource = it->second.lock();
					if (resource != nullptr)
					{
						serializable_count++;
					}
				}

				// Write all embedded resources:
				archive << serializable_count;
				for (auto& name : resource_names)
				{
					auto it = resources.find(name);
					if (it == resources.end())
						continue;
					wi::allocator::shared_ptr<ResourceInternal> resource = it->second.lock();

					if (resource != nullptr)
					{
						std::string name = it->first;
						wi::helper::MakePathRelative(archive.GetSourceDirectory(), name);

						if (resource->filedata.empty())
						{
							// Directly re-read the file part that is needed:
							wi::helper::FileRead(
								resource->container_filename,
								resource->filedata,
								resource->container_filesize,
								resource->container_fileoffset
							);
						}

						archive << name;
						archive << (uint32_t)resource->flags;
						archive << resource->filedata;

						if (!archive.GetSourceFileName().empty())
						{
							// Refresh the container file properties to the current file:
							//	The old file offsets could get stale otherwise if it's overwritten
							resource->container_filename = archive.GetSourceFileName();
							resource->container_fileoffset = archive.GetPos() - resource->filedata.size();
							resource->container_filesize = resource->filedata.size();
							if (archive.IsCompressionEnabled())
							{
								// Compressed archive: retain file data to keep resource streamable
								resource->flags |= Flags::IMPORT_RETAIN_FILEDATA;
							}
							if (!has_flag(resource->flags, Flags::IMPORT_RETAIN_FILEDATA))
							{
								resource->filedata.clear();
								resource->filedata.shrink_to_fit();
							}
						}
					}
				}
			}
			locker.unlock();
		}

	}

}
