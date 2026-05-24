using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using CvSize = OpenCvSharp.Size;
using CvPoint = OpenCvSharp.Point;

namespace AnomalyDetection.Api.Services
{
    internal static class ForegroundMask
    {
        private const int ClosingRadius = 5;
        private const int OpeningRadius = 2;

        public static bool[,] Compute(Image<Rgb24> image, bool applyMask)
        {
            int h = image.Height;
            int w = image.Width;

            if (!applyMask)
            {
                var allOnes = new bool[h, w];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        allOnes[y, x] = true;
                return allOnes;
            }

            using var gray = ToGrayMat(image);

            using var blurred = new Mat();
            Cv2.GaussianBlur(gray, blurred, new CvSize(5, 5), 0);

            using var thresh = new Mat();
            Cv2.Threshold(blurred, thresh, 0, 255,
                ThresholdTypes.Binary | ThresholdTypes.Otsu);

            int margin = (int)(Math.Min(h, w) * 0.05);

            double borderMedianThresh = ComputeBorderMedian(thresh, h, w, margin);

            if (borderMedianThresh > 127.0)
                Cv2.BitwiseNot(thresh, thresh);

            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int numLabels = Cv2.ConnectedComponentsWithStats(
                thresh, labels, stats, centroids,
                PixelConnectivity.Connectivity8);

            using var componentMask = new Mat(h, w, MatType.CV_8UC1, Scalar.Black);

            if (numLabels > 1)
            {
                int largestLabel = 1;
                int largestArea = 0;
                for (int lbl = 1; lbl < numLabels; lbl++)
                {
                    int area = stats.At<int>(lbl, (int)ConnectedComponentsTypes.Area);
                    if (area > largestArea)
                    {
                        largestArea = area;
                        largestLabel = lbl;
                    }
                }

                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        if (labels.At<int>(y, x) == largestLabel)
                            componentMask.Set(y, x, (byte)255);
            }
            else
            {
                thresh.CopyTo(componentMask);
            }

            FillHoles(componentMask);

            using var closingKernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new CvSize(2 * ClosingRadius + 1, 2 * ClosingRadius + 1));
            Cv2.MorphologyEx(componentMask, componentMask, MorphTypes.Close, closingKernel);

            FillHoles(componentMask);

            using var openingKernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new CvSize(2 * OpeningRadius + 1, 2 * OpeningRadius + 1));
            Cv2.MorphologyEx(componentMask, componentMask, MorphTypes.Open, openingKernel);

            return MatToBoolMask(componentMask, h, w);
        }

        private static Mat ToGrayMat(Image<Rgb24> image)
        {
            int h = image.Height;
            int w = image.Width;
            var mat = new Mat(h, w, MatType.CV_8UC1);

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < h; y++)
                {
                    Span<Rgb24> row = accessor.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        byte luma = (byte)(
                            0.299f * row[x].R +
                            0.587f * row[x].G +
                            0.114f * row[x].B);
                        mat.Set(y, x, luma);
                    }
                }
            });

            return mat;
        }

        private static double ComputeBorderMedian(Mat mat, int h, int w, int margin)
        {
            var borderValues = new List<byte>(2 * h * w);

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (y < margin || y >= h - margin || x < margin || x >= w - margin)
                        borderValues.Add(mat.At<byte>(y, x));

            if (borderValues.Count == 0) return 0.0;

            borderValues.Sort();
            int mid = borderValues.Count / 2;
            return borderValues.Count % 2 == 0
                ? (borderValues[mid - 1] + borderValues[mid]) / 2.0
                : borderValues[mid];
        }

        private static void FillHoles(Mat mask)
        {
            int h = mask.Rows;
            int w = mask.Cols;

            using var inv = new Mat();
            Cv2.BitwiseNot(mask, inv);

            using var floodMask = Mat.Zeros(h + 2, w + 2, MatType.CV_8UC1).ToMat();
            Cv2.FloodFill(inv, floodMask, new CvPoint(0, 0), new Scalar(128));
            Cv2.FloodFill(inv, floodMask, new CvPoint(w - 1, 0), new Scalar(128));
            Cv2.FloodFill(inv, floodMask, new CvPoint(0, h - 1), new Scalar(128));
            Cv2.FloodFill(inv, floodMask, new CvPoint(w - 1, h - 1), new Scalar(128));

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (inv.At<byte>(y, x) == 255)
                        mask.Set(y, x, (byte)255);
        }

        private static bool[,] MatToBoolMask(Mat mat, int h, int w)
        {
            var result = new bool[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    result[y, x] = mat.At<byte>(y, x) > 0;
            return result;
        }
    }
}