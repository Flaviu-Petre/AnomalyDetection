using anomaly_detection_app.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
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
                    _inferenceService?.Dispose();
                    SelectedModelPath = openFileDialog.FileName;
                    _inferenceService = new AnomalyDetectionService(SelectedModelPath);

                    ResultText = "Model loaded. Now please load the corresponding Metadata JSON.";
                }
                catch (Exception ex)
                {
                    ResultText = $"Error loading model: {ex.Message}";
                    SelectedModelPath = string.Empty;
                }
            }
        }

        [RelayCommand]
        private async Task SelectMetadataAsync()
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "JSON Files|*.json",
                Title = "Select Metadata JSON File"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string jsonString = await File.ReadAllTextAsync(openFileDialog.FileName);
                    var metadata = JsonSerializer.Deserialize<ModelMetadata>(jsonString);

                    if (metadata != null)
                    {
                        _anomalyThreshold = metadata.Threshold;

                        MetadataInfo = $"Category: {metadata.Category.ToUpper()} | Model: {metadata.ModelName} | Threshold: {_anomalyThreshold:F4}";
                        ResultText = "Metadata loaded successfully. Now insert an image to inspect.";
                    }
                }
                catch (Exception ex)
                {
                    ResultText = $"Error loading metadata: {ex.Message}";
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
                var result = await Task.Run(() => _inferenceService.PredictAnomalyScore(SelectedImagePath));

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
