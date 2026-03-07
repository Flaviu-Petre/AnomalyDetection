using anomaly_detection_app.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Size = SixLabors.ImageSharp.Size;

public class AnomalyDetectionService : IDisposable
{
    private readonly InferenceSession _padimSession;
    private readonly InferenceSession _yoloSession;

    public AnomalyDetectionService(string padimModelPath, string yoloModelPath)
    {
        // Initialize both ONNX sessions
        _padimSession = new InferenceSession(padimModelPath);
        _yoloSession = new InferenceSession(yoloModelPath);
    }

    public AnomalyResult PredictAnomalyScore(string imagePath, bool applyMask = false)
    {
        // 1. Load Image
        using var image = SixLabors.ImageSharp.Image.Load<Rgba32>(imagePath);

        // 2. Preprocess: Resize to 256x256, Center Crop to 224x224
        image.Mutate(x => x
            .Resize(new ResizeOptions
            {
                Size = new Size(256, 256),
                Mode = ResizeMode.Stretch,
                Sampler = KnownResamplers.Triangle
            })
            .Crop(new Rectangle(16, 16, 224, 224))
        );

        // 3. Convert to Tensor and Normalize (ImageNet Standards for PaDiM)
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var std = new[] { 0.229f, 0.224f, 0.225f };

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < pixelRow.Length; x++)
                {
                    ref Rgba32 pixel = ref pixelRow[x];

                    float normR = ((pixel.R / 255f) - mean[0]) / std[0];
                    float normG = ((pixel.G / 255f) - mean[1]) / std[1];
                    float normB = ((pixel.B / 255f) - mean[2]) / std[2];

                    inputTensor[0, 0, y, x] = normR;
                    inputTensor[0, 1, y, x] = normG;
                    inputTensor[0, 2, y, x] = normB;
                }
            }
        });

        // 4. Run Inference through PaDiM
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _padimSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        float[] rawMap = outputTensor.ToArray();

        // 5. DYNAMIC BACKGROUND MASK 
        if (applyMask)
        {
            float[] yoloMask = GenerateYoloMask(image);

            for (int i = 0; i < rawMap.Length; i++)
            {
                if (yoloMask[i] == 0f)
                {
                    rawMap[i] = 0f;
                }
            }
        }

        // 6. Global Score Calculation 
        float maxScore = 0f;
        int borderOffset = 16;

        for (int y = borderOffset; y < 224 - borderOffset; y++)
        {
            for (int x = borderOffset; x < 224 - borderOffset; x++)
            {
                float val = rawMap[y * 224 + x];
                if (val > maxScore)
                {
                    maxScore = val;
                }
            }
        }

        // 7. ADVANCED POST-PROCESSING & MORPHOLOGY 
        using Mat rawMat = new Mat(224, 224, MatType.CV_32FC1);
        rawMat.SetArray(rawMap);

        using Mat binaryTopologyMat = RefineAnomalyTopology(rawMat);

        byte[] topologyBytes = new byte[224 * 224];
        binaryTopologyMat.GetArray(out topologyBytes);

        // 8. Generate the Defect Overlay Image
        using var heatmap = new SixLabors.ImageSharp.Image<Rgba32>(224, 224);
        heatmap.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < 224; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                int rowOffset = y * 224;

                for (int x = 0; x < 224; x++)
                {
                    if (topologyBytes[rowOffset + x] == 255)
                    {
                        pixelRow[x] = new Rgba32(255, 0, 0, 150);
                    }
                    else
                    {
                        pixelRow[x] = new Rgba32(0, 0, 0, 0);
                    }
                }
            }
        });

        image.Mutate(ctx => ctx.DrawImage(heatmap, PixelColorBlendingMode.Normal, PixelAlphaCompositionMode.SrcOver, 1.0f));

        using var ms = new MemoryStream();
        image.SaveAsJpeg(ms);

        return new AnomalyResult
        {
            Score = maxScore,
            HeatmapImageBytes = ms.ToArray()
        };
    }

    private Mat RefineAnomalyTopology(Mat rawMahalanobisHeatmap)
    {
        // 1. Normalize the raw floating-point anomaly distances into an 8-bit grayscale matrix (0-255)
        using Mat normalizedMap = new Mat();
        Cv2.Normalize(rawMahalanobisHeatmap, normalizedMap, 0, 255, NormTypes.MinMax, (int)MatType.CV_8UC1);

        // 2. Execute Adaptive Thresholding (Mean-C) to binarize the heatmap based on local illumination gradients.
        using Mat thresholdedMap = new Mat();
        Cv2.AdaptiveThreshold(
            normalizedMap,
            thresholdedMap,
            255,
            AdaptiveThresholdTypes.MeanC,
            ThresholdTypes.Binary,
            15,
            -2
        );

        // 3. Define the structuring element (kernel) for the ensuing Morphological Operations.
        using Mat structuringElement = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5));

        // 4. Morphological Opening: Systematically annihilate isolated micro-dust particles and sensor noise.
        using Mat openedMap = new Mat();
        Cv2.MorphologyEx(thresholdedMap, openedMap, MorphTypes.Open, structuringElement);

        // 5. Morphological Closing: Bridge the gaps between fragmented defect segments to form contiguous geometries.
        Mat finalDefectTopology = new Mat();
        Cv2.MorphologyEx(openedMap, finalDefectTopology, MorphTypes.Close, structuringElement);

        return finalDefectTopology;
    }

    private float[] GenerateYoloMask(SixLabors.ImageSharp.Image<Rgba32> sourceImage)
    {
        // 1. Preprocess for YOLOv8 (Standard 640x640 input)
        using var yoloImage = sourceImage.Clone(x => x.Resize(new ResizeOptions
        {
            Size = new Size(640, 640),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Triangle
        }));

        var inputTensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
        yoloImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgba32> row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    inputTensor[0, 0, y, x] = row[x].R / 255f;
                    inputTensor[0, 1, y, x] = row[x].G / 255f;
                    inputTensor[0, 2, y, x] = row[x].B / 255f;
                }
            }
        });

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
        using var results = _yoloSession.Run(inputs);

        // 2. Extract YOLOv8-Seg Output Tensors
        var output0 = results.First(v => v.Name == "output0").AsTensor<float>(); 
        var output1 = results.First(v => v.Name == "output1").AsTensor<float>(); 

        // 3. Find the best detection 
        int bestAnchor = 0;
        float maxConf = 0;
        for (int i = 0; i < 8400; i++)
        {
            float conf = output0[0, 4, i];
            if (conf > maxConf)
            {
                maxConf = conf;
                bestAnchor = i;
            }
        }

        // 4. Matrix Multiplication: Coefficients * Prototypes
        float[] mask160 = new float[160 * 160];

        if (maxConf > 0.50f)
        {
            float[] coeffs = new float[32];
            for (int c = 0; c < 32; c++) coeffs[c] = output0[0, 5 + c, bestAnchor];

            for (int y = 0; y < 160; y++)
            {
                for (int x = 0; x < 160; x++)
                {
                    float val = 0;
                    for (int c = 0; c < 32; c++)
                    {
                        val += coeffs[c] * output1[0, c, y, x];
                    }

                    float sigmoid = 1.0f / (1.0f + (float)Math.Exp(-val));

                    mask160[y * 160 + x] = sigmoid > 0.5f ? 1.0f : 0.0f;
                }
            }
        }

        // 5. Resize the 160x160 binary mask back to PaDiM's 224x224 coordinate space using ImageSharp
        using var maskImage = new SixLabors.ImageSharp.Image<L8>(160, 160);
        maskImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < 160; y++)
            {
                Span<L8> row = accessor.GetRowSpan(y);
                for (int x = 0; x < 160; x++)
                {
                    row[x] = new L8((byte)(mask160[y * 160 + x] * 255));
                }
            }
        });

        maskImage.Mutate(x => x.Resize(224, 224, KnownResamplers.NearestNeighbor));

        // 6. Output the final 224x224 binary filter
        float[] finalMask224 = new float[224 * 224];
        maskImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < 224; y++)
            {
                Span<L8> row = accessor.GetRowSpan(y);
                for (int x = 0; x < 224; x++)
                {
                    finalMask224[y * 224 + x] = row[x].PackedValue > 127 ? 1.0f : 0.0f;
                }
            }
        });

        return finalMask224;
    }

    public void Dispose()
    {
        _padimSession?.Dispose();
        _yoloSession?.Dispose();
    }
}