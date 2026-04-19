using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Text.Json;
using Size = SixLabors.ImageSharp.Size;

namespace AnomalyDetection.Api.Services
{
    public class RouterService : IDisposable
    {
        #region Fields
        private readonly InferenceSession _routerSession;
        private readonly Dictionary<string, string> _classMapping;
        private readonly ILogger<RouterService> _logger;
        #endregion 

        #region Constructor
        public RouterService(ILogger<RouterService> logger)
        {
            _logger = logger;

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "RouterModel", "router.onnx");
            string jsonPath = Path.Combine(Directory.GetCurrentDirectory(), "RouterModel", "classes.json");

            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Router ONNX model not found at: {modelPath}");

            if (!File.Exists(jsonPath))
                throw new FileNotFoundException($"Router class mapping not found at: {jsonPath}");

            _logger.LogInformation("[ROUTER] Loading router model into memory from: {Path}", modelPath);

            var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _routerSession = new InferenceSession(modelPath, options);

            string jsonContent = File.ReadAllText(jsonPath);
            _classMapping = JsonSerializer.Deserialize<Dictionary<string, string>>(jsonContent)
                ?? throw new InvalidOperationException("Failed to deserialize classes.json.");

            _logger.LogInformation("[ROUTER] Router model loaded. Known categories: {Categories}",
                string.Join(", ", _classMapping.Values));
        }
        #endregion

        #region Public Methods
        public async Task<(string Category, float Confidence)> ClassifyAsync(Stream imageStream)
        {
            imageStream.Position = 0;

            using var image = await Image.LoadAsync<Rgb24>(imageStream);

            int w = image.Width;
            int h = image.Height;
            int newW = w < h ? 256 : (int)Math.Round(256.0 * w / h);
            int newH = h < w ? 256 : (int)Math.Round(256.0 * h / w);

            image.Mutate(x => x
                .Resize(new ResizeOptions
                {
                    Size = new Size(newW, newH),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Bicubic
                })
            );

            int cropX = (image.Width - 224) / 2;
            int cropY = (image.Height - 224) / 2;
            image.Mutate(x => x.Crop(new Rectangle(cropX, cropY, 224, 224)));

            var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> row = accessor.GetRowSpan(y);
                    for (int x = 0; x < row.Length; x++)
                    {
                        tensor[0, 0, y, x] = ((row[x].R / 255f) - mean[0]) / std[0];
                        tensor[0, 1, y, x] = ((row[x].G / 255f) - mean[1]) / std[1];
                        tensor[0, 2, y, x] = ((row[x].B / 255f) - mean[2]) / std[2];
                    }
                }
            });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = _routerSession.Run(inputs);
            float[] logits = results.First().AsEnumerable<float>().ToArray();

            float maxLogit = logits.Max();
            float sumExp = logits.Sum(l => (float)Math.Exp(l - maxLogit));

            int bestIndex = 0;
            float bestConfidence = 0f;

            for (int i = 0; i < logits.Length; i++)
            {
                float probability = (float)Math.Exp(logits[i] - maxLogit) / sumExp;
                if (probability > bestConfidence)
                {
                    bestConfidence = probability;
                    bestIndex = i;
                }
            }

            string predictedCategory = _classMapping[bestIndex.ToString()];

            _logger.LogDebug("[ROUTER] Predicted category: {Category} ({Confidence:P1})",
                predictedCategory, bestConfidence);

            imageStream.Position = 0;

            return (predictedCategory, bestConfidence);
        }

        public void Dispose()
        {
            _routerSession.Dispose();
        }
        #endregion
    }
}
