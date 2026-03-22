namespace AnomalyDetection.Api.Services
{
    public class FeedbackService
    {
        #region Fields
        private readonly string _baseFeedbackDirectory = "FeedbackData";
        #endregion

        #region Methods
        public async Task<string> SaveFeedbackImageAsync(string category, bool isActuallyAnomaly, IFormFile image)
        {
            string normalizedCategory = category.ToLower().Trim();

            string labelFolder = isActuallyAnomaly ? "anomaly" : "good";

            string directoryPath = Path.Combine(Directory.GetCurrentDirectory(), _baseFeedbackDirectory, normalizedCategory, labelFolder);

            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            string extension = Path.GetExtension(image.FileName);
            if (string.IsNullOrEmpty(extension)) extension = ".png";

            string uniqueFileName = $"{DateTime.UtcNow:yyyyMMdd_HHmmssfff}_{Guid.NewGuid().ToString().Substring(0, 6)}{extension}";
            string fullFilePath = Path.Combine(directoryPath, uniqueFileName);

            using (var stream = new FileStream(fullFilePath, FileMode.Create))
            {
                await image.CopyToAsync(stream);
            }

            return fullFilePath;
        }
        #endregion
    }
}
