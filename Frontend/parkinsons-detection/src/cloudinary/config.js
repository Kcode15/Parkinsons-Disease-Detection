// Cloudinary configuration for browser usage
export const CLOUDINARY_CONFIG = {
  cloud_name: 'dqqckzxhl',
  api_key: '473556569339723',
  api_secret: '50HlR7YSDnKnGYlOQVlM-ARLPCU'
};

// Upload function using fetch API instead of Cloudinary SDK
export const uploadToCloudinary = async (file, uploadPreset) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('upload_preset', uploadPreset);
  formData.append('cloud_name', CLOUDINARY_CONFIG.cloud_name);

  try {
    const response = await fetch(
      `https://api.cloudinary.com/v1_1/${CLOUDINARY_CONFIG.cloud_name}/${
        file.type.startsWith('image/') ? 'image' : 'video'
      }/upload`,
      {
        method: 'POST',
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Cloudinary upload error:', error);
    throw error;
  }
};
