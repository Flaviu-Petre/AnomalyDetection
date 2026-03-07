using anomaly_detection_app.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OpenCvSharp;
using Microsoft.Win32;
using System.Windows;
using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace anomaly_detection_app.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private AnomalyDetectionService? _inferenceService;
        private float _anomalyThreshold = 10.0f;

        [ObservableProperty]
        private string _selectedImagePath;

        [ObservableProperty]
        private string _selectedModelPath;

        [ObservableProperty]
        private string _metadataInfo;

        [ObservableProperty]
        private BitmapImage _heatmapImageSource;

        [ObservableProperty]
        private bool _isObjectCategory;

        [ObservableProperty]
        private string _resultText;

        [ObservableProperty]
        private bool _isBusy;

        public MainViewModel()
        {
            ResultText = "Step 1: Load ONNX Model. Step 2: Load JSON Metadata.";
            MetadataInfo = "No metadata loaded.";
        }

        [RelayCommand]
        private void SelectModel()
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "ONNX Model Files|*.onnx",
                Title = "Select Anomaly Detection Model"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string padimPath = openFileDialog.FileName;
                    string directory = Path.GetDirectoryName(padimPath);

                    string yoloPath = Path.Combine(directory, "yolov8n-seg.onnx");

                    if (!File.Exists(yoloPath))
                    {
                        MessageBox.Show("Could not find 'yolov8n-seg.onnx' in the same directory as the PaDiM model.\n\nPlease ensure both ONNX files are in the same folder before loading.",
                                        "Missing YOLO Model",
                                        MessageBoxButton.OK,
                                        MessageBoxImage.Warning);
                        return;
                    }

                    _inferenceService?.Dispose();
                    SelectedModelPath = padimPath;

                    _inferenceService = new AnomalyDetectionService(SelectedModelPath, yoloPath);

                    ResultText = "PaDiM and YOLOv8 models loaded successfully. Now please load the corresponding Metadata JSON.";
                }
                catch (Exception ex)
                {
                    ResultText = $"Error loading models: {ex.Message}";
                    SelectedModelPath = string.Empty;
                }
            }
        }

        [RelayCommand]
        private async Task SelectMetadataAsync()
        {
            if (_inferenceService == null)
            {
                ResultText = "Error: Please load the ONNX Models first before loading Metadata.";
                return;
            }

            var openFileDialog = new OpenFileDialog
            {
                Filter = "JSON Files|*.json",
                Title = "Select Metadata JSON File"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                IsBusy = true;
                ResultText = "Calibrating threshold...";

                try
                {
                    string jsonPath = openFileDialog.FileName;
                    string jsonString = await File.ReadAllTextAsync(jsonPath);
                    var metadata = JsonSerializer.Deserialize<ModelMetadata>(jsonString);

                    if (metadata != null)
                    {
                        string directory = Path.GetDirectoryName(jsonPath);
                        string calibrationImagePath = Path.Combine(directory, "calibration_image.png");

                        IsObjectCategory = metadata.Category.ToLower() != "carpet" &&
                                           metadata.Category.ToLower() != "grid" &&
                                           metadata.Category.ToLower() != "leather" &&
                                           metadata.Category.ToLower() != "tile" &&
                                           metadata.Category.ToLower() != "wood";

                        if (File.Exists(calibrationImagePath))
                        {
                            var calibrationResult = await Task.Run(() => _inferenceService.PredictAnomalyScore(calibrationImagePath, IsObjectCategory));

                            _anomalyThreshold = calibrationResult.Score * 1.30f;

                            MetadataInfo = $"Category: {metadata.Category.ToUpper()} | INT8 Threshold: {_anomalyThreshold:F2}";
                            ResultText = "Calibration loaded successfully. Now insert an image to inspect.";
                        }
                        else
                        {
                            _anomalyThreshold = metadata.Threshold;
                            MetadataInfo = $"Category: {metadata.Category.ToUpper()} | Threshold: {_anomalyThreshold:F2} (Uncalibrated)";
                            ResultText = "Warning: 'calibration_image.png' not found in JSON folder. Using uncalibrated Python threshold.";
                        }
                    }
                }
                catch (Exception ex)
                {
                    ResultText = $"Error loading metadata: {ex.Message}";
                }
                finally
                {
                    IsBusy = false;
                }
            }
        }

        [RelayCommand]
        private void SelectImage()
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Select Image to Analyze"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                SelectedImagePath = openFileDialog.FileName;
                ResultText = "Image loaded. Click 'Run Inference'.";
            }
        }

        [RelayCommand]
        private async Task RunInferenceAsync()
        {
            if (_inferenceService == null || string.IsNullOrEmpty(SelectedImagePath)) return;

            IsBusy = true;
            ResultText = "Analyzing...";

            try
            {
                var result = await Task.Run(() => _inferenceService.PredictAnomalyScore(SelectedImagePath, IsObjectCategory));

                string status = result.Score > _anomalyThreshold ? "ANOMALY DETECTED" : "NORMAL";
                ResultText = $"Status: {status}\nMax Anomaly Score: {result.Score:F4} \n(Threshold was {_anomalyThreshold:F4})";

                Application.Current.Dispatcher.Invoke(() =>
                {
                    var bitmap = new BitmapImage();
                    using (var mem = new MemoryStream(result.HeatmapImageBytes))
                    {
                        mem.Position = 0;
                        bitmap.BeginInit();
                        bitmap.CacheOption = BitmapCacheOption.OnLoad;
                        bitmap.StreamSource = mem;
                        bitmap.EndInit();
                    }
                    bitmap.Freeze();
                    HeatmapImageSource = bitmap;
                });
            }
            catch (Exception ex)
            {
                ResultText = $"Error: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}