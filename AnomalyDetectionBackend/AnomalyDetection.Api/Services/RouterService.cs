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
        #region Constants
        private static readonly float[] ClipMean = { 0.48145466f, 0.4578275f, 0.40821073f };
        private static readonly float[] ClipStd = { 0.26862954f, 0.26130258f, 0.27577711f };
        private const int EmbeddingDim = 512;
        private const int ImageSize = 224;
        private const float TemperatureScale = 100f;

        private static readonly HashSet<string> PerCategoryThresholdClasses = new()
        {
            "toothbrush",   
            "tile",        
            "metal_nut",   
        };
        #endregion

        #region Fields
        private readonly InferenceSession _clipSession;
        private readonly float[,] _textEmbeddings;
        private readonly List<string> _categories;
        private readonly float _oodThreshold;
        private readonly Dictionary<string, string> _decoyRemap;
        private readonly Dictionary<string, float> _perCategoryThresholds;
        private readonly ILogger<RouterService> _logger;
        #endregion

        #region Constructor
        public RouterService(ILogger<RouterService> logger)
        {
            _logger = logger;

            string encoderPath = Path.Combine(Directory.GetCurrentDirectory(), "Router", "clip_image_encoder.onnx");
            string embeddingsPath = Path.Combine(Directory.GetCurrentDirectory(), "Router", "text_embeddings.npy");
            string configPath = Path.Combine(Directory.GetCurrentDirectory(), "Router", "clip_router_config.json");

            if (!File.Exists(encoderPath))
                throw new FileNotFoundException($"CLIP encoder not found at: {encoderPath}");
            if (!File.Exists(embeddingsPath))
                throw new FileNotFoundException($"Text embeddings not found at: {embeddingsPath}");
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"Router config not found at: {configPath}");

            var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _clipSession = new InferenceSession(encoderPath, options);

            var config = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(configPath));
            _oodThreshold = config.GetProperty("global_ood_threshold").GetSingle();

            _categories = LoadCategories(config);
            _textEmbeddings = LoadNpy(embeddingsPath, _categories.Count, EmbeddingDim);

            _decoyRemap = new Dictionary<string, string>();
            if (config.TryGetProperty("decoy_remap", out var remapEl))
            {
                foreach (var kv in remapEl.EnumerateObject())
                    _decoyRemap[kv.Name] = kv.Value.GetString()!;
            }

            _perCategoryThresholds = new Dictionary<string, float>();
            if (config.TryGetProperty("per_category_thresholds", out var perCatEl))
            {
                foreach (var kv in perCatEl.EnumerateObject())
                    if (PerCategoryThresholdClasses.Contains(kv.Name))
                        _perCategoryThresholds[kv.Name] = kv.Value.GetSingle();
            }

            _logger.LogInformation("[CLIP ROUTER] Loaded. Categories: {Categories}",
                string.Join(", ", _categories));
            _logger.LogInformation("[CLIP ROUTER] Decoy remap: {Remap}",
                string.Join(", ", _decoyRemap.Select(kv => $"{kv.Key}→{kv.Value}")));
            _logger.LogInformation("[CLIP ROUTER] Confidence gating: ACTIVE (capsule/pill<0.91, bottle<0.75, carpet<0.90)");
            _logger.LogInformation("[CLIP ROUTER] Per-category thresholds active for: {Classes}",
                string.Join(", ", _perCategoryThresholds.Select(kv => $"{kv.Key}={kv.Value:P1}")));
        }
        #endregion

        #region Public Methods
        public async Task<(string Category, float Confidence)> ClassifyAsync(Stream imageStream)
        {
            imageStream.Position = 0;

            using var image = await LoadAndPreprocessAsync(imageStream);
            var tensor = BuildInputTensor(image);
            float[] imageEmbedding = RunEncoder(tensor);
            float[] probs = ComputeSoftmax(imageEmbedding);

            int bestIndex = Array.IndexOf(probs, probs.Max());
            float confidence = probs[bestIndex];
            string category = _categories[bestIndex];

            if (_decoyRemap.TryGetValue(category, out var realCategory))
            {
                _logger.LogDebug("[CLIP ROUTER] Decoy '{Decoy}' remapped to '{Real}'",
                    category, realCategory);
                category = realCategory;
            }

            (category, confidence) = ApplyConfidenceGating(category, confidence, probs);

            float effectiveThreshold = _perCategoryThresholds.TryGetValue(category, out var catThreshold)
                ? catThreshold
                : _oodThreshold;

            if (confidence < effectiveThreshold)
            {
                _logger.LogWarning(
                    "[CLIP ROUTER] OOD rejected. Category: {Category}, Confidence: {Confidence:P1}, " +
                    "Threshold: {Threshold:P1} ({ThresholdType})",
                    category, confidence, effectiveThreshold,
                    _perCategoryThresholds.ContainsKey(category) ? "per-category" : "global");
                return ("unknown", confidence);
            }

            _logger.LogDebug("[CLIP ROUTER] Predicted: {Category} ({Confidence:P1})", category, confidence);

            imageStream.Position = 0;
            return (category, confidence);
        }

        public void Dispose()
        {
            _clipSession.Dispose();
        }
        #endregion

        #region Pipeline Steps
        private static async Task<Image<Rgb24>> LoadAndPreprocessAsync(Stream imageStream)
        {
            var image = await Image.LoadAsync<Rgb24>(imageStream);
            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(ImageSize, ImageSize),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Bicubic
            }));
            return image;
        }

        private static DenseTensor<float> BuildInputTensor(Image<Rgb24> image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> row = accessor.GetRowSpan(y);
                    for (int x = 0; x < row.Length; x++)
                    {
                        tensor[0, 0, y, x] = ((row[x].R / 255f) - ClipMean[0]) / ClipStd[0];
                        tensor[0, 1, y, x] = ((row[x].G / 255f) - ClipMean[1]) / ClipStd[1];
                        tensor[0, 2, y, x] = ((row[x].B / 255f) - ClipMean[2]) / ClipStd[2];
                    }
                }
            });
            return tensor;
        }

        private float[] RunEncoder(DenseTensor<float> tensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", tensor)
            };

            using var results = _clipSession.Run(inputs);
            return results.First().AsEnumerable<float>().ToArray();
        }

        private float[] ComputeSoftmax(float[] imageEmbedding)
        {
            int numCategories = _categories.Count;
            float[] logits = new float[numCategories];

            for (int i = 0; i < numCategories; i++)
            {
                float dot = 0f;
                for (int j = 0; j < EmbeddingDim; j++)
                    dot += imageEmbedding[j] * _textEmbeddings[i, j];
                logits[i] = TemperatureScale * dot;
            }

            float maxLogit = logits.Max();
            float sumExp = logits.Sum(l => (float)Math.Exp(l - maxLogit));

            return logits.Select(l => (float)Math.Exp(l - maxLogit) / sumExp).ToArray();
        }
        #endregion

        #region Private Helpers

        private (string Category, float Confidence) ApplyConfidenceGating(
            string category, float confidence, float[] probs)
        {
            if ((category == "capsule" || category == "pill") && confidence < 0.905f)
            {
                _logger.LogInformation(
                    "[CLIP ROUTER] Confidence gating: '{Category}' ({Confidence:P1}) → 'toothbrush'",
                    category, confidence);
                return ("toothbrush", confidence);
            }

            if (category == "bottle" && confidence < 0.75f)
            {
                int idx = _categories.IndexOf("metal_nut");
                if (idx >= 0 && probs[idx] > 0.10f)
                {
                    float combined = confidence + probs[idx];
                    _logger.LogInformation(
                        "[CLIP ROUTER] Confidence gating: '{Category}' ({Confidence:P1}) → 'metal_nut' " +
                        "(metal_nut score: {Score:P1}, combined: {Combined:P1})",
                        category, confidence, probs[idx], combined);
                    return ("metal_nut", combined);
                }
            }

            if (category == "carpet" && confidence < 0.90f)
            {
                int idx = _categories.IndexOf("tile");
                if (idx >= 0 && probs[idx] > 0.05f)
                {
                    float combined = confidence + probs[idx];
                    _logger.LogInformation(
                        "[CLIP ROUTER] Confidence gating: '{Category}' ({Confidence:P1}) → 'tile' " +
                        "(tile score: {Score:P1}, combined: {Combined:P1})",
                        category, confidence, probs[idx], combined);
                    return ("tile", combined);
                }
            }

            return (category, confidence);
        }

        private static List<string> LoadCategories(JsonElement config)
        {
            var categoriesObj = config.GetProperty("categories");

            var entries = new SortedDictionary<int, string>();
            foreach (var prop in categoriesObj.EnumerateObject())
                entries[int.Parse(prop.Name)] = prop.Value.GetString()!;

            return entries.Values.ToList();
        }

        private static float[,] LoadNpy(string path, int rows, int cols)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            br.ReadBytes(8);
            ushort headerLen = br.ReadUInt16();
            br.ReadBytes(headerLen);

            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = br.ReadSingle();

            return result;
        }
        #endregion
    }
}