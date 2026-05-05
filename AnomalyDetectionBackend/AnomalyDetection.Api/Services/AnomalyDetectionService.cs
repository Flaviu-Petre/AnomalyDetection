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
        private const int GridSize = 16;
        private const int FeatureDim = 768;
        private const double GaussianSigma = 4.0;

        private const float OverlayImageWeight = 0.6f;
        private const float OverlayHeatmapWeight = 0.4f;

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
        public AnomalyResult PredictAnomalyScore(Stream imageStream, float threshold, float scoreMin, float scoreMax, bool returnHeatmap = false)
        {
            using var image = LoadAndPreprocess(imageStream);
            var tensor = BuildInputTensor(image);
            float[,] feats = RunEncoder(tensor);
            float[,] scores = ComputePatchScores(feats);
            float[,] map = UpsampleScoreMap(scores);
            float[,] smooth = GaussianSmooth(map, GaussianSigma);
            float rawScore = Percentile(smooth, 99.5f);

            float denom = (scoreMax - scoreMin) + 1e-8f;
            float normalized = (rawScore - scoreMin) / denom;
            if (normalized < 0f) normalized = 0f;

            bool isAnomaly = normalized > threshold;

            _logger.LogDebug(
                "[PATCHCORE] raw={Raw:F4}  min={Min:F4}  max={Max:F4}  norm={Norm:F4}  thr={Thr:F4}  anomaly={IsAnomaly}",
                rawScore, scoreMin, scoreMax, normalized, threshold, isAnomaly);

            string? heatmap = returnHeatmap ? GenerateOverlayHeatmapBase64(smooth, image) : null;

            return new AnomalyResult
            {
                IsAnomaly = isAnomaly,
                Score = normalized,
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

        private static string GenerateOverlayHeatmapBase64(float[,] map, Image<Rgb24> input)
        {
            int size = map.GetLength(0);

            float max = float.MinValue;
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    if (map[y, x] > max) max = map[y, x];

            float denom = max + 1e-8f;

            int width = Math.Min(input.Width, size);
            int height = Math.Min(input.Height, size);

            using var output = new Image<Rgb24>(width, height);

            input.ProcessPixelRows(output, (inAcc, outAcc) =>
            {
                for (int y = 0; y < height; y++)
                {
                    Span<Rgb24> inRow = inAcc.GetRowSpan(y);
                    Span<Rgb24> outRow = outAcc.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        float t = map[y, x] / denom;
                        if (t < 0f) t = 0f;
                        else if (t > 1f) t = 1f;

                        Rgb24 jet = JetColormap(t);

                        float r = OverlayImageWeight * (inRow[x].R / 255f) + OverlayHeatmapWeight * (jet.R / 255f);
                        float g = OverlayImageWeight * (inRow[x].G / 255f) + OverlayHeatmapWeight * (jet.G / 255f);
                        float b = OverlayImageWeight * (inRow[x].B / 255f) + OverlayHeatmapWeight * (jet.B / 255f);

                        outRow[x] = new Rgb24(
                            (byte)Math.Clamp(r * 255f, 0f, 255f),
                            (byte)Math.Clamp(g * 255f, 0f, 255f),
                            (byte)Math.Clamp(b * 255f, 0f, 255f));
                    }
                }
            });

            using var ms = new MemoryStream();
            output.SaveAsPng(ms);
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