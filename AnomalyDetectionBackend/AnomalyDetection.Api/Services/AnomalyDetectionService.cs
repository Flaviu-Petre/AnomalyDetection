using AnomalyDetection.Api.Models.Domain;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
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
        public AnomalyResult PredictAnomalyScore(Stream imageStream, float threshold, float scoreMin, float scoreMax, bool applyMask, bool heatmapUseGlobalMax, bool returnHeatmap = false)
        {
            using var image = LoadAndPreprocess(imageStream);
            var tensor = BuildInputTensor(image);
            float[,] feats = RunEncoder(tensor);
            float[,] scores = ComputePatchScores(feats);
            float[,] smooth = UpsampleAndSmooth(scores, ImageSize, GaussianSigma);

            bool[,] mask = ForegroundMask.Compute(image, applyMask);
            ApplyMask(smooth, mask);

            float rawScore = Percentile(smooth, mask, 99.5f);

            float denom = (scoreMax - scoreMin) + 1e-8f;
            float normalized = (rawScore - scoreMin) / denom;
            if (normalized < 0f) normalized = 0f;

            bool isAnomaly = normalized > threshold;

            _logger.LogDebug(
                "[PATCHCORE] raw={Raw:F4}  min={Min:F4}  max={Max:F4}  norm={Norm:F4}  thr={Thr:F4}  anomaly={IsAnomaly}  masking={ApplyMask}",
                rawScore, scoreMin, scoreMax, normalized, threshold, isAnomaly, applyMask);

            string? heatmap = returnHeatmap ? GenerateOverlayHeatmapBase64(smooth, mask, image, scoreMin, scoreMax, heatmapUseGlobalMax, isAnomaly) : null;

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

            Parallel.For(0, numPatches, p =>
            {
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

                Array.Sort(distances);

                float meanDist = 0f;
                for (int k = 0; k < _kNeighbours; k++)
                    meanDist += distances[k];
                meanDist /= _kNeighbours;

                int row = p / GridSize;
                int col = p % GridSize;

                scores[row, col] = meanDist;
            });

            return scores;
        }

        private static float[,] UpsampleAndSmooth(float[,] scores, int targetSize, double sigma)
        {
            int gridSize = scores.GetLength(0);

            using var smallMat = new Mat(gridSize, gridSize, MatType.CV_32FC1);
            for (int y = 0; y < gridSize; y++)
                for (int x = 0; x < gridSize; x++)
                    smallMat.Set(y, x, scores[y, x]);

            using var resizedMat = new Mat();
            Cv2.Resize(smallMat, resizedMat, new OpenCvSharp.Size(targetSize, targetSize), 0, 0, InterpolationFlags.Nearest);

            int radius = (int)(3 * sigma);
            int kernelSize = 2 * radius + 1;
            using var blurredMat = new Mat();
            Cv2.GaussianBlur(resizedMat, blurredMat, new OpenCvSharp.Size(kernelSize, kernelSize), sigma, sigma);

            var result = new float[targetSize, targetSize];
            for (int y = 0; y < targetSize; y++)
                for (int x = 0; x < targetSize; x++)
                    result[y, x] = blurredMat.At<float>(y, x);

            return result;
        }

        private static void ApplyMask(float[,] map, bool[,] mask)
        {
            int size = map.GetLength(0);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    if (!mask[y, x])
                        map[y, x] = 0f;
        }

        private static float Percentile(float[,] map, bool[,] mask, float percentile)
        {
            int size = map.GetLength(0);

            var values = new List<float>(size * size);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    if (mask[y, x])
                        values.Add(map[y, x]);

            if (values.Count == 0)
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size; x++)
                        values.Add(map[y, x]);

            values.Sort();
            float rank = (percentile / 100f) * (values.Count - 1);
            int lower = (int)rank;
            int upper = Math.Min(lower + 1, values.Count - 1);
            float frac = rank - lower;
            return values[lower] + frac * (values[upper] - values[lower]);
        }

        private static (float Lo, float Hi) CalculateHeatmapBounds( float[,] map, bool[,] mask, float scoreMin, float scoreMax, bool heatmapUseGlobalMax, bool isAnomaly)
        {
            if (heatmapUseGlobalMax && !isAnomaly)
            {
                return (0f, scoreMax);
            }

            int size = map.GetLength(0);
            var mapValues = new List<float>(size * size);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    if (mask[y, x])
                    {
                        mapValues.Add(map[y, x]);
                    }
                }
            }

            if (mapValues.Count == 0)
            {
                return (scoreMin, scoreMax);
            }

            mapValues.Sort();
            int n = mapValues.Count;
            float lo = mapValues[(int)(0.01f * (n - 1))];
            float hi = mapValues[(int)(0.99f * (n - 1))];

            return (lo, hi);
        }

        private static string GenerateOverlayHeatmapBase64( float[,] map, bool[,] mask, Image<Rgb24> input, float scoreMin, float scoreMax, bool heatmapUseGlobalMax, bool isAnomaly)
        {
            var (lo, hi) = CalculateHeatmapBounds(map, mask, scoreMin, scoreMax, heatmapUseGlobalMax, isAnomaly);
            float denom = (hi - lo) + 1e-8f;

            int width = Math.Min(input.Width, map.GetLength(1));
            int height = Math.Min(input.Height, map.GetLength(0));

            using var output = new Image<Rgb24>(width, height);

            input.ProcessPixelRows(output, (inAcc, outAcc) =>
            {
                for (int y = 0; y < height; y++)
                {
                    Span<Rgb24> inRow = inAcc.GetRowSpan(y);
                    Span<Rgb24> outRow = outAcc.GetRowSpan(y);

                    for (int x = 0; x < width; x++)
                    {
                        if (!mask[y, x])
                        {
                            outRow[x] = inRow[x];
                            continue;
                        }

                        float t = Math.Clamp((map[y, x] - lo) / denom, 0f, 1f);

                        Rgb24 jet = JetColormap(t);

                        float r = (OverlayImageWeight * inRow[x].R) + (OverlayHeatmapWeight * jet.R);
                        float g = (OverlayImageWeight * inRow[x].G) + (OverlayHeatmapWeight * jet.G);
                        float b = (OverlayImageWeight * inRow[x].B) + (OverlayHeatmapWeight * jet.B);

                        outRow[x] = new Rgb24(
                            (byte)Math.Clamp(r, 0f, 255f),
                            (byte)Math.Clamp(g, 0f, 255f),
                            (byte)Math.Clamp(b, 0f, 255f)
                        );
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