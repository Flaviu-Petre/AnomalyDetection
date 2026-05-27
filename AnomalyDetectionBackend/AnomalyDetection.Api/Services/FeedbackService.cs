using AnomalyDetection.Api.Services.Interfaces;

namespace AnomalyDetection.Api.Services
{
    public class FeedbackService : IFeedbackService
    {
        #region Fields
        private readonly string _baseFeedbackDirectory = "FeedbackData";
        private readonly ILogger<FeedbackService> _logger;
        #endregion

        #region Constructor
        public FeedbackService(ILogger<FeedbackService> logger)
        {
            _logger = logger;
        }
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
                _logger.LogInformation("[FILE SYSTEM] Created new feedback directory at: {DirectoryPath}", directoryPath);
            }

            string extension = Path.GetExtension(image.FileName);
            if (string.IsNullOrEmpty(extension)) extension = ".png";

            string uniqueFileName = $"{DateTime.UtcNow:yyyyMMdd_HHmmssfff}_{Guid.NewGuid().ToString().Substring(0, 6)}{extension}";
            string fullFilePath = Path.Combine(directoryPath, uniqueFileName);

            using (var stream = new FileStream(fullFilePath, FileMode.Create))
            {
                await image.CopyToAsync(stream);
            }

            _logger.LogInformation("[FILE SYSTEM] Saved user feedback image: {FilePath}", fullFilePath);

            return fullFilePath;
        }

        public List<object> GetFeedbackSummary()
        {
            var result = new List<object>();
            string basePath = Path.Combine(Directory.GetCurrentDirectory(), _baseFeedbackDirectory);

            if (!Directory.Exists(basePath))
                return result;

            foreach (var categoryDir in Directory.GetDirectories(basePath))
            {
                string categoryName = Path.GetFileName(categoryDir);
                int anomalyCount = CountFiles(Path.Combine(categoryDir, "anomaly"));
                int goodCount = CountFiles(Path.Combine(categoryDir, "good"));

                if (anomalyCount == 0 && goodCount == 0)
                    continue;

                result.Add(new
                {
                    Category = categoryName,
                    AnomalyCount = anomalyCount,
                    GoodCount = goodCount
                });
            }

            return result;
        }

        public List<string> GetFeedbackImageNames(string category, string label)
        {
            string dirPath = Path.Combine(Directory.GetCurrentDirectory(), _baseFeedbackDirectory,
                category.ToLower().Trim(), label);

            if (!Directory.Exists(dirPath))
                return new List<string>();

            return Directory.GetFiles(dirPath)
                .Select(Path.GetFileName)
                .Where(f => f != null)
                .ToList()!;
        }

        public (Stream stream, string contentType) GetFeedbackImageStream(string category, string label, string filename)
        {
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), _baseFeedbackDirectory,
                category.ToLower().Trim(), label, filename);

            if (!File.Exists(filePath))
                throw new FileNotFoundException();

            string ext = Path.GetExtension(filename).ToLower();
            string contentType = ext switch
            {
                ".png" => "image/png",
                ".jpg" => "image/jpeg",
                ".jpeg" => "image/jpeg",
                ".bmp" => "image/bmp",
                _ => "application/octet-stream"
            };

            return (File.OpenRead(filePath), contentType);
        }

        private int CountFiles(string dirPath) =>
            Directory.Exists(dirPath) ? Directory.GetFiles(dirPath).Length : 0;
        #endregion
    }
}
