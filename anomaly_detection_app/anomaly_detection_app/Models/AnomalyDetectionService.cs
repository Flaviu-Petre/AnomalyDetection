using anomaly_detection_app.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class AnomalyDetectionService : IDisposable
{
    private readonly InferenceSession _session;

    public AnomalyDetectionService(string modelPath)
    {
        // Initialize the ONNX session
        _session = new InferenceSession(modelPath);
    }

    public AnomalyResult PredictAnomalyScore(string imagePath, bool applyMask = false)
    {
        // 1. Load Image
        using var image = Image.Load<Rgba32>(imagePath);

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

        // 3. Convert to Tensor, Normalize, and Extract Brightness
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var std = new[] { 0.229f, 0.224f, 0.225f };

        float[] brightnessMap = new float[224 * 224];

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                int rowOffset = y * 224;

                for (int x = 0; x < pixelRow.Length; x++)
                {
                    ref Rgba32 pixel = ref pixelRow[x];

                    float normR = ((pixel.R / 255f) - mean[0]) / std[0];
                    float normG = ((pixel.G / 255f) - mean[1]) / std[1];
                    float normB = ((pixel.B / 255f) - mean[2]) / std[2];

                    inputTensor[0, 0, y, x] = normR;
                    inputTensor[0, 1, y, x] = normG;
                    inputTensor[0, 2, y, x] = normB;

                    brightnessMap[rowOffset + x] = (normR + normG + normB) / 3f;
                }
            }
        });

        // 4. Run Inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // 5. Apply Gaussian Blur 
        float[] smoothedMap = ApplyGaussianBlur(outputTensor.ToArray(), 224, 224, 8f);

        // --- 6. DYNAMIC BACKGROUND MASK ---
        if (applyMask)
        {
            float bMin = brightnessMap.Min();
            float bMax = brightnessMap.Max();
            float thresh = bMin + 0.15f * (bMax - bMin); 

            float[] rawMask = new float[224 * 224];
            for (int i = 0; i < rawMask.Length; i++)
            {
                rawMask[i] = brightnessMap[i] > thresh ? 1.0f : 0.0f;
            }

            float[] smoothedMask = ApplyGaussianBlur(rawMask, 224, 224, 5f);

            for (int i = 0; i < smoothedMap.Length; i++)
            {
                if (smoothedMask[i] <= 0.5f)
                {
                    smoothedMap[i] = 0f;
                }
            }
        }

        // 7. Find Max Score for the final anomaly reading
        float maxScore = smoothedMap.Max();
        float minScore = smoothedMap.Min();

        // 8. Generate the Heatmap Image
        using var heatmap = new Image<Rgba32>(224, 224);
        heatmap.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < 224; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                int rowOffset = y * 224;

                for (int x = 0; x < 224; x++)
                {
                    float v = smoothedMap[rowOffset + x];
                    float normalized = (v - minScore) / (maxScore - minScore + 1e-5f);

                    float r = Math.Clamp(1.5f - Math.Abs(4.0f * normalized - 3.0f), 0f, 1f);
                    float g = Math.Clamp(1.5f - Math.Abs(4.0f * normalized - 2.0f), 0f, 1f);
                    float b = Math.Clamp(1.5f - Math.Abs(4.0f * normalized - 1.0f), 0f, 1f);

                    pixelRow[x] = new Rgba32((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 150);
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

    private float[] ApplyGaussianBlur(float[] map, int width = 224, int height = 224, float sigma = 8f)
    {
        int radius = (int)Math.Ceiling(4 * sigma);
        int size = 2 * radius + 1;
        float[] kernel = new float[size];
        float sum = 0;

        for (int i = 0; i < size; i++)
        {
            int x = i - radius;
            kernel[i] = (float)Math.Exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;

        float[] temp = new float[height * width];
        float[] finalMap = new float[height * width];

        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++)
            {
                float val = 0;
                for (int i = 0; i < size; i++)
                {
                    int px = GetReflectIndex(x + i - radius, width);
                    val += map[rowOffset + px] * kernel[i];
                }
                temp[rowOffset + x] = val;
            }
        }

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                float val = 0;
                for (int i = 0; i < size; i++)
                {
                    int py = GetReflectIndex(y + i - radius, height);
                    val += temp[py * width + x] * kernel[i];
                }
                finalMap[y * width + x] = val;
            }
        }

        return finalMap;
    }

    private int GetReflectIndex(int index, int max)
    {
        if (index < 0) return -index - 1;
        if (index >= max) return 2 * max - index - 1;
        return index;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}