using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System;
using System.Threading.Tasks;

namespace anomaly_detection_app.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private AnomalyDetectionService? _inferenceService;

        [ObservableProperty]
        private string _selectedImagePath;

        [ObservableProperty]
        private string _selectedModelPath;

        [ObservableProperty]
        private string _resultText;

        [ObservableProperty]
        private bool _isBusy;

        public MainViewModel()
        {
            ResultText = "Please load an ONNX model.";
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

                    ResultText = "Model loaded successfully. Now insert an image.";
                }
                catch (Exception ex)
                {
                    ResultText = $"Error loading model: {ex.Message}";
                    SelectedModelPath = string.Empty;
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

                if (_inferenceService == null)
                    ResultText = "Image loaded. Please load an ONNX model first.";
                else
                    ResultText = "Image loaded. Click 'Run Inference'.";
            }

            try
            {
                // Run inference off the UI thread
                float score = await Task.Run(() => _inferenceService.PredictAnomalyScore(SelectedImagePath));

                // You can adjust this threshold based on your model's evaluation
                string status = score > 10.0f ? "ANOMALY DETECTED" : "NORMAL";
                ResultText = $"Status: {status}\nMax Anomaly Score: {score:F4}";
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
