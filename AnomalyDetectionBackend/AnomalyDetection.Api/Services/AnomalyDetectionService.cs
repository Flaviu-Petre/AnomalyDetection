using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using AnomalyDetection.Api.Models.Domain;
using Size = SixLabors.ImageSharp.Size;

namespace AnomalyDetection.Api.Services
{
    public class AnomalyDetectionService : IDisposable
    {
        #region Constants
        private const int ImageSize = 224;
        private const int ResizeSize = 256;
        private const int GridSize = 16;
        private const int FeatureDim = 768;
        private const double GaussianSigma = 4.0;

        private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };
        #endregion

        #region Fields
        private readonly InferenceSession _encoderSession;
        private readonly float[,] _memoryBank;
        private readonly int _kNeighbours;
        private readonly ILogger<AnomalyDetectionService> _logger;
        #endregion

        #region Constructor
        public AnomalyDetectionService(
            string encoderPath,
            string bankPath,
            int kNeighbours,
            ILogger<AnomalyDetectionService> logger)
        {
            _logger = logger;
            _kNeighbours = kNeighbours;

            var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _encoderSession = new InferenceSession(encoderPath, options);

            _memoryBank = LoadMemoryBank(bankPath);

            _logger.LogInformation("[PATCHCORE] Loaded memory bank: {Vectors} vectors of dim {Dim}",
                _memoryBank.GetLength(0), _memoryBank.GetLength(1));
        }
        #endregion

        #region Public Methods
        public AnomalyResult PredictAnomalyScore(Stream imageStream, float threshold, bool returnHeatmap = false)
        {
            using var image = LoadAndPreprocess(imageStream);
            var tensor = BuildInputTensor(image);
            float[,] feats = RunEncoder(tensor);
            float[,] scores = ComputePatchScores(feats);
            float[,] map = UpsampleScoreMap(scores);
            float[,] smooth = GaussianSmooth(map, GaussianSigma);
            float rawScore = Percentile(smooth, 99.5f);
            float score = rawScore;

            bool isAnomaly = score > threshold;
            string? heatmap = returnHeatmap ? GenerateHeatmapBase64(smooth, image) : null;

            return new AnomalyResult
            {
                IsAnomaly = isAnomaly,
                Score = score,
                UsedThreshold = threshold,
                HeatmapBase64 = heatmap
            };
        }

        public void Dispose()
        {
            _encoderSession.Dispose();
        }
        #endregion

        #region Pipeline Steps
        private static Image<Rgb24> LoadAndPreprocess(Stream imageStream)
        {
            var image = Image.Load<Rgb24>(imageStream);
            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(ResizeSize, ResizeSize),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Bicubic
            }));

            int cropX = (image.Width - ImageSize) / 2;
            int cropY = (image.Height - ImageSize) / 2;
            image.Mutate(x => x.Crop(new Rectangle(cropX, cropY, ImageSize, ImageSize)));

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
                        tensor[0, 0, y, x] = ((row[x].R / 255f) - ImageNetMean[0]) / ImageNetStd[0];
                        tensor[0, 1, y, x] = ((row[x].G / 255f) - ImageNetMean[1]) / ImageNetStd[1];
                        tensor[0, 2, y, x] = ((row[x].B / 255f) - ImageNetMean[2]) / ImageNetStd[2];
                    }
                }
            });
            return tensor;
        }

        private float[,] RunEncoder(DenseTensor<float> tensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", tensor)
            };

            using var results = _encoderSession.Run(inputs);
            float[] raw = results.First().AsEnumerable<float>().ToArray();

            var patches = new float[GridSize * GridSize, FeatureDim];
            for (int c = 0; c < FeatureDim; c++)
                for (int h = 0; h < GridSize; h++)
                    for (int w = 0; w < GridSize; w++)
                        patches[h * GridSize + w, c] = raw[c * GridSize * GridSize + h * GridSize + w];

            return patches;
        }

        private float[,] ComputePatchScores(float[,] patches)
        {
            int numPatches = patches.GetLength(0);
            int bankSize = _memoryBank.GetLength(0);
            var scores = new float[GridSize, GridSize];

            for (int p = 0; p < numPatches; p++)
            {
                // Find k nearest neighbours in memory bank
                var distances = new float[bankSize];
                for (int b = 0; b < bankSize; b++)
                {
                    float dist = 0f;
                    for (int d = 0; d < FeatureDim; d++)
                    {
                        float diff = patches[p, d] - _memoryBank[b, d];
                        dist += diff * diff;
                    }
                    distances[b] = dist;
                }

                // Mean of k smallest distances
                Array.Sort(distances);
                float meanDist = 0f;
                for (int k = 0; k < _kNeighbours; k++)
                    meanDist += distances[k];
                meanDist /= _kNeighbours;

                int row = p / GridSize;
                int col = p % GridSize;
                scores[row, col] = meanDist;
            }

            return scores;
        }

        private static float[,] UpsampleScoreMap(float[,] scores)
        {
            var upsampled = new float[ImageSize, ImageSize];
            float scaleH = (float)GridSize / ImageSize;
            float scaleW = (float)GridSize / ImageSize;

            for (int y = 0; y < ImageSize; y++)
                for (int x = 0; x < ImageSize; x++)
                {
                    int srcY = Math.Min((int)(y * scaleH), GridSize - 1);
                    int srcX = Math.Min((int)(x * scaleW), GridSize - 1);
                    upsampled[y, x] = scores[srcY, srcX];
                }

            return upsampled;
        }

        private static float[,] GaussianSmooth(float[,] map, double sigma)
        {
            int size = map.GetLength(0);
            var result = new float[size, size];
            int radius = (int)(3 * sigma);
            double twoSigmaSq = 2 * sigma * sigma;

            // Build 1D kernel
            int kernelSize = 2 * radius + 1;
            var kernel = new double[kernelSize];
            double sum = 0;
            for (int i = 0; i < kernelSize; i++)
            {
                int x = i - radius;
                kernel[i] = Math.Exp(-(x * x) / twoSigmaSq);
                sum += kernel[i];
            }
            for (int i = 0; i < kernelSize; i++)
                kernel[i] /= sum;

            // Horizontal pass
            var temp = new float[size, size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    double val = 0;
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int sx = Math.Clamp(x + k - radius, 0, size - 1);
                        val += map[y, sx] * kernel[k];
                    }
                    temp[y, x] = (float)val;
                }

            // Vertical pass
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    double val = 0;
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int sy = Math.Clamp(y + k - radius, 0, size - 1);
                        val += temp[sy, x] * kernel[k];
                    }
                    result[y, x] = (float)val;
                }

            return result;
        }

        private static float Percentile(float[,] map, float percentile)
        {
            int size = map.GetLength(0);
            var flat = new float[size * size];
            int idx = 0;
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    flat[idx++] = map[y, x];

            Array.Sort(flat);
            float rank = (percentile / 100f) * (flat.Length - 1);
            int lower = (int)rank;
            int upper = Math.Min(lower + 1, flat.Length - 1);
            float frac = rank - lower;
            return flat[lower] + frac * (flat[upper] - flat[lower]);
        }

        private static string GenerateHeatmapBase64(float[,] map, Image<Rgb24> original)
        {
            int size = map.GetLength(0);
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    if (map[y, x] < min) min = map[y, x];
                    if (map[y, x] > max) max = map[y, x];
                }

            float range = max - min + 1e-8f;

            using var heatmap = new Image<Rgb24>(size, size);
            heatmap.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < size; y++)
                {
                    Span<Rgb24> row = accessor.GetRowSpan(y);
                    for (int x = 0; x < size; x++)
                    {
                        float norm = (map[y, x] - min) / range;
                        row[x] = JetColormap(norm);
                    }
                }
            });

            using var ms = new MemoryStream();
            heatmap.SaveAsPng(ms);
            return Convert.ToBase64String(ms.ToArray());
        }

        private static Rgb24 JetColormap(float t)
        {
            float r = Math.Clamp(1.5f - Math.Abs(4f * t - 3f), 0f, 1f);
            float g = Math.Clamp(1.5f - Math.Abs(4f * t - 2f), 0f, 1f);
            float b = Math.Clamp(1.5f - Math.Abs(4f * t - 1f), 0f, 1f);
            return new Rgb24((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
        }
        #endregion

        #region Private Helpers
        private static float[,] LoadMemoryBank(string path)
        {
            using var zip = System.IO.Compression.ZipFile.OpenRead(path);
            var entry = zip.GetEntry("memory_bank.npy")
                ?? throw new InvalidOperationException("memory_bank.npy not found in .npz file");

            using var stream = entry.Open();
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            ms.Position = 0;

            using var br = new BinaryReader(ms);
            br.ReadBytes(8);
            ushort headerLen = br.ReadUInt16();
            string header = System.Text.Encoding.ASCII.GetString(br.ReadBytes(headerLen));

            var shapeMatch = System.Text.RegularExpressions.Regex.Match(header, @"'shape':\s*\((\d+),\s*(\d+)\)");
            int rows = int.Parse(shapeMatch.Groups[1].Value);
            int cols = int.Parse(shapeMatch.Groups[2].Value);

            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = br.ReadSingle();

            return result;
        }
        #endregion
    }
}