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
        private const int ResizeSize = 256;
        private const int InputSize = 224;
        private const int CropOffset = 16;

        private static readonly float[] NormMean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] NormStd = { 0.229f, 0.224f, 0.225f };

        private const int SmoothingKernelSize = 31;
        private const double SmoothingSigma = 8.0;

        private const int BorderOffset = 8;
        private const double RobustPercentile = 0.995;

        private const double ContrastMaskThreshold = 15.0;
        #endregion

        #region Fields
        private readonly InferenceSession _padimSession;
        private readonly ILogger<AnomalyDetectionService> _logger;
        #endregion

        #region Constructor
        public AnomalyDetectionService(string padimModelPath, ILogger<AnomalyDetectionService> logger)
        {
            _logger = logger;
            var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _logger.LogInformation("[ONNX] Loading Padim model into memory from: {Path}", padimModelPath);
            _padimSession = new InferenceSession(padimModelPath, options);
            
        }
        #endregion

        #region Public methods
        public AnomalyResult PredictAnomalyScore(Stream imageStream, float threshold, bool applyMask = false, bool returnHeatmap = false)
        {
            _logger.LogDebug("[ONNX] Starting Padim inference execution...");

            using var image = LoadAndPreprocess(imageStream);
            var inputTensor = NormalizeToTensor(image);
            float[] rawMap = RunInference(inputTensor);
            float[] smoothedMap = SmoothMap(rawMap);

            if (applyMask)
            {
                ApplyContrastMask(smoothedMap, image);
            }

            float score = ComputeRobustScore(smoothedMap);
            string? heatmap = returnHeatmap ? GenerateHeatmapBase64(smoothedMap, image) : null;

            return new AnomalyResult
            {
                IsAnomaly = score > threshold,
                Score = score,
                UsedThreshold = threshold,
                HeatmapBase64 = heatmap
            };
        }

        public void Dispose()
        {
            _padimSession.Dispose();
        }
        #endregion

        #region Pipeline Steps
        private static Image<Rgba32> LoadAndPreprocess(Stream imageStream)
        {
            var image = Image.Load<Rgba32>(imageStream);

            image.Mutate(x => x
                .Resize(new ResizeOptions
                {
                    Size = new Size(ResizeSize, ResizeSize),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Bicubic
                })
                .Crop(new Rectangle(CropOffset, CropOffset, InputSize, InputSize))
            );

            return image;
        }

        private static DenseTensor<float> NormalizeToTensor(Image<Rgba32> image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < pixelRow.Length; x++)
                    {
                        ref Rgba32 pixel = ref pixelRow[x];
                        tensor[0, 0, y, x] = ((pixel.R / 255f) - NormMean[0]) / NormStd[0];
                        tensor[0, 1, y, x] = ((pixel.G / 255f) - NormMean[1]) / NormStd[1];
                        tensor[0, 2, y, x] = ((pixel.B / 255f) - NormMean[2]) / NormStd[2];
                    }
                }
            });

            return tensor;
        }

        private float[] RunInference(DenseTensor<float> inputTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _padimSession.Run(inputs);
            return results.First().AsTensor<float>().ToArray();
        }

        private static float[] SmoothMap(float[] rawMap)
        {
            using Mat rawMat = new Mat(InputSize, InputSize, MatType.CV_32FC1);
            rawMat.SetArray(rawMap);

            using Mat blurredMat = new Mat();
            Cv2.GaussianBlur(rawMat, blurredMat, new OpenCvSharp.Size(SmoothingKernelSize, SmoothingKernelSize), SmoothingSigma);

            blurredMat.GetArray(out float[] blurredMap);
            return blurredMap;
        }

        private static void ApplyContrastMask(float[] map, Image<Rgba32> image)
        {
            float[] contrastMask = GenerateContrastMask(image);
            for (int i = 0; i < map.Length; i++)
            {
                if (contrastMask[i] == 0f)
                {
                    map[i] = 0f;
                }
            }
        }

        private static float ComputeRobustScore(float[] map)
        {
            var validScores = new List<float>();

            for (int y = BorderOffset; y < InputSize - BorderOffset; y++)
            {
                for (int x = BorderOffset; x < InputSize - BorderOffset; x++)
                {
                    float val = map[y * InputSize + x];
                    if (val > 0) validScores.Add(val);
                }
            }

            if (validScores.Count == 0) return 0f;

            validScores.Sort();
            int index = (int)(validScores.Count * RobustPercentile);
            return validScores[Math.Min(index, validScores.Count - 1)];
        }
        #endregion

        #region Heatmap & mask generation
        private static string GenerateHeatmapBase64(float[] blurredMap, Image<Rgba32> image)
        {
            using Mat maskedBlurredMat = new Mat(InputSize, InputSize, MatType.CV_32FC1);
            maskedBlurredMat.SetArray(blurredMap);

            using Mat normalizedMap = new Mat();
            Cv2.Normalize(maskedBlurredMat, normalizedMap, 0, 255, NormTypes.MinMax, (int)MatType.CV_8UC1);

            using Mat colorMap = new Mat();
            Cv2.ApplyColorMap(normalizedMap, colorMap, ColormapTypes.Jet);

            using var heatmapOverlay = new Image<Rgba32>(InputSize, InputSize);
            heatmapOverlay.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < InputSize; y++)
                {
                    Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < InputSize; x++)
                    {
                        Vec3b color = colorMap.At<Vec3b>(y, x);
                        byte alpha = 128;
                        pixelRow[x] = new Rgba32(color.Item2, color.Item1, color.Item0, alpha);
                    }
                }
            });

            image.Mutate(ctx => ctx.DrawImage(heatmapOverlay, PixelColorBlendingMode.Normal, PixelAlphaCompositionMode.SrcOver, 1.0f));

            using var ms = new MemoryStream();
            image.SaveAsPng(ms);

            return Convert.ToBase64String(ms.ToArray());
        }

        private static float[] GenerateContrastMask(Image<Rgba32> image)
        {
            using Mat grayMat = new Mat(InputSize, InputSize, MatType.CV_8UC1);

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < InputSize; y++)
                {
                    Span<Rgba32> row = accessor.GetRowSpan(y);
                    for (int x = 0; x < InputSize; x++)
                    {
                        byte gray = (byte)(0.299 * row[x].R + 0.587 * row[x].G + 0.114 * row[x].B);
                        grayMat.Set<byte>(y, x, gray);
                    }
                }
            });

            using Mat blurred = new Mat();
            Cv2.GaussianBlur(grayMat, blurred, new OpenCvSharp.Size(5, 5), 0);

            using Mat bgRoi = new Mat(blurred, new OpenCvSharp.Rect(0, 0, 10, 10));
            Scalar bgMean = Cv2.Mean(bgRoi);

            using Mat diffMat = new Mat();
            Cv2.Absdiff(blurred, new Scalar(bgMean.Val0), diffMat);

            using Mat threshMat = new Mat();
            Cv2.Threshold(diffMat, threshMat, ContrastMaskThreshold, 255, ThresholdTypes.Binary);

            using Mat openKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(5, 5));
            using Mat openedMat = new Mat();
            Cv2.MorphologyEx(threshMat, openedMat, MorphTypes.Open, openKernel);

            Cv2.FindContours(openedMat, out OpenCvSharp.Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            using Mat maskMat = new Mat(InputSize, InputSize, MatType.CV_8UC1, new Scalar(0));

            if (contours.Length > 0)
            {
                double maxArea = 0;
                int maxAreaIdx = -1;
                for (int i = 0; i < contours.Length; i++)
                {
                    double area = Cv2.ContourArea(contours[i]);
                    if (area > maxArea) { maxArea = area; maxAreaIdx = i; }
                }

                if (maxAreaIdx != -1) Cv2.DrawContours(maskMat, contours, maxAreaIdx, new Scalar(255), Cv2.FILLED);
                else maskMat.SetTo(new Scalar(255));
            }
            else
            {
                maskMat.SetTo(new Scalar(255));
            }

            using Mat finalMaskMat = new Mat();
            using Mat dilateKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(15, 15));
            Cv2.Dilate(maskMat, finalMaskMat, dilateKernel);

            float[] finalMask = new float[InputSize * InputSize];
            finalMaskMat.GetArray(out byte[] maskBytes);

            for (int i = 0; i < maskBytes.Length; i++)
            {
                finalMask[i] = maskBytes[i] > 127 ? 1.0f : 0.0f;
            }

            return finalMask;
        }
        #endregion
    }
}
