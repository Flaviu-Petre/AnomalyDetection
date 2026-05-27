using AnomalyDetection.Api.Services;
using FluentAssertions;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Moq;

namespace AnomalyDetection.Tests.Unit.Services
{
    public class FeedbackServiceTests : IDisposable
    {
        #region Setup
        private readonly Mock<ILogger<FeedbackService>> _mockLogger;
        private readonly FeedbackService _feedbackService;
        private readonly string _testBaseDirectory;

        public FeedbackServiceTests()
        {
            _mockLogger = new Mock<ILogger<FeedbackService>>();
            _feedbackService = new FeedbackService(_mockLogger.Object);

            _testBaseDirectory = Path.Combine(Directory.GetCurrentDirectory(), "FeedbackData");
        }

        public void Dispose()
        {
            if (Directory.Exists(_testBaseDirectory))
                Directory.Delete(_testBaseDirectory, recursive: true);
        }
        #endregion

        #region Helpers
        private static IFormFile CreateMockImageFile(string fileName = "test.png", string content = "fake-image-content")
        {
            var mockFile = new Mock<IFormFile>();
            var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(content));

            mockFile.Setup(f => f.FileName).Returns(fileName);
            mockFile.Setup(f => f.Length).Returns(stream.Length);
            mockFile.Setup(f => f.CopyToAsync(It.IsAny<Stream>(), It.IsAny<CancellationToken>()))
                    .Callback<Stream, CancellationToken>((s, _) => stream.CopyTo(s))
                    .Returns(Task.CompletedTask);

            return mockFile.Object;
        }
        #endregion

        #region SaveFeedbackImageAsync Tests
        [Fact]
        public async Task SaveFeedbackImageAsync_ReturnsFilePath_WhenImageIsAnomaly()
        {
            // Arrange
            var mockFile = CreateMockImageFile("anomaly.png");

            // Act
            var result = await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);

            // Assert
            result.Should().NotBeNullOrEmpty();
            result.Should().Contain("bottle");
            result.Should().Contain("anomaly");
        }

        [Fact]
        public async Task SaveFeedbackImageAsync_ReturnsFilePath_WhenImageIsGood()
        {
            // Arrange
            var mockFile = CreateMockImageFile("good.png");

            // Act
            var result = await _feedbackService.SaveFeedbackImageAsync("bottle", false, mockFile);

            // Assert
            result.Should().NotBeNullOrEmpty();
            result.Should().Contain("bottle");
            result.Should().Contain("good");
        }

        [Fact]
        public async Task SaveFeedbackImageAsync_CreatesDirectory_WhenItDoesNotExist()
        {
            // Arrange
            var mockFile = CreateMockImageFile("test.png");
            string expectedDir = Path.Combine(Directory.GetCurrentDirectory(), "FeedbackData", "capsule", "anomaly");

            // Act
            await _feedbackService.SaveFeedbackImageAsync("capsule", true, mockFile);

            // Assert
            Directory.Exists(expectedDir).Should().BeTrue();
        }

        [Fact]
        public async Task SaveFeedbackImageAsync_NormalizesCategory_ToLowercase()
        {
            // Arrange
            var mockFile = CreateMockImageFile("test.png");

            // Act
            var result = await _feedbackService.SaveFeedbackImageAsync("BOTTLE", true, mockFile);

            // Assert
            result.Should().Contain("bottle");
            result.Should().NotContain("BOTTLE");
        }

        [Fact]
        public async Task SaveFeedbackImageAsync_UsesDefaultExtension_WhenFileHasNoExtension()
        {
            // Arrange
            var mockFile = CreateMockImageFile("imagewithoutextension");

            // Act
            var result = await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);

            // Assert
            result.Should().EndWith(".png");
        }

        [Fact]
        public async Task SaveFeedbackImageAsync_PreservesExtension_WhenFileHasJpgExtension()
        {
            // Arrange
            var mockFile = CreateMockImageFile("photo.jpg");

            // Act
            var result = await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);

            // Assert
            result.Should().EndWith(".jpg");
        }

        [Fact]
        public async Task SaveFeedbackImageAsync_GeneratesUniqueFileNames_ForMultipleSaves()
        {
            // Arrange
            var mockFile1 = CreateMockImageFile("test.png");
            var mockFile2 = CreateMockImageFile("test.png");

            // Act
            var result1 = await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile1);
            await Task.Delay(5); // Asigura timestamp diferit
            var result2 = await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile2);

            // Assert
            result1.Should().NotBe(result2);
        }
        #endregion

        #region GetFeedbackSummary Tests
        [Fact]
        public void GetFeedbackSummary_ReturnsEmptyList_WhenNoFeedbackExists()
        {
            // Act
            var result = _feedbackService.GetFeedbackSummary();

            // Assert
            result.Should().NotBeNull();
            result.Should().BeEmpty();
        }

        [Fact]
        public async Task GetFeedbackSummary_ReturnsSummary_AfterFeedbackIsSaved()
        {
            // Arrange
            var mockFile = CreateMockImageFile("test.png");
            await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);

            // Act
            var result = _feedbackService.GetFeedbackSummary();

            // Assert
            result.Should().NotBeEmpty();
            result.Should().HaveCount(1);
        }

        [Fact]
        public void GetFeedbackSummary_ReturnsEmptyList_WhenDirectoryIsEmpty()
        {
            // Arrange 
            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), "FeedbackData", "bottle", "anomaly"));

            // Act
            var result = _feedbackService.GetFeedbackSummary();

            // Assert
            result.Should().BeEmpty();
        }
        #endregion

        #region GetFeedbackImageNames Tests
        [Fact]
        public void GetFeedbackImageNames_ReturnsEmptyList_WhenDirectoryDoesNotExist()
        {
            // Act
            var result = _feedbackService.GetFeedbackImageNames("nonexistent", "anomaly");

            // Assert
            result.Should().NotBeNull();
            result.Should().BeEmpty();
        }

        [Fact]
        public async Task GetFeedbackImageNames_ReturnsFileNames_AfterSaving()
        {
            // Arrange
            var mockFile = CreateMockImageFile("test.png");
            await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);

            // Act
            var result = _feedbackService.GetFeedbackImageNames("bottle", "anomaly");

            // Assert
            result.Should().NotBeEmpty();
            result.Should().HaveCount(1);
        }

        [Fact]
        public void GetFeedbackImageNames_NormalizesCategory_ToLowercase()
        {
            // Act
            var result = _feedbackService.GetFeedbackImageNames("BOTTLE", "anomaly");

            // Assert 
            result.Should().NotBeNull();
            result.Should().BeEmpty();
        }
        #endregion

        #region GetFeedbackImageStream Tests
        [Fact]
        public async Task GetFeedbackImageStream_ReturnsStream_WhenFileExists()
        {
            // Arrange
            var mockFile = CreateMockImageFile("test.png");
            await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);
            var fileNames = _feedbackService.GetFeedbackImageNames("bottle", "anomaly");
            var fileName = fileNames.First();

            // Act
            var (stream, contentType) = _feedbackService.GetFeedbackImageStream("bottle", "anomaly", fileName);

            // Assert
            stream.Should().NotBeNull();
            contentType.Should().Be("image/png");
            stream.Dispose();
        }

        [Fact]
        public void GetFeedbackImageStream_ThrowsFileNotFoundException_WhenFileDoesNotExist()
        {
            // Act
            Action act = () => _feedbackService.GetFeedbackImageStream("bottle", "anomaly", "nonexistent.png");

            // Assert
            act.Should().Throw<FileNotFoundException>();
        }

        [Theory]
        [InlineData("image.png", "image/png")]
        [InlineData("image.jpg", "image/jpeg")]
        [InlineData("image.jpeg", "image/jpeg")]
        [InlineData("image.bmp", "image/bmp")]
        [InlineData("image.xyz", "application/octet-stream")]
        public async Task GetFeedbackImageStream_ReturnsCorrectContentType_ForExtension(string fileName, string expectedContentType)
        {
            // Arrange
            var mockFile = CreateMockImageFile(fileName);
            await _feedbackService.SaveFeedbackImageAsync("bottle", true, mockFile);
            var fileNames = _feedbackService.GetFeedbackImageNames("bottle", "anomaly");
            var savedFileName = fileNames.First();

            // Act
            var (stream, contentType) = _feedbackService.GetFeedbackImageStream("bottle", "anomaly", savedFileName);

            // Assert
            contentType.Should().Be(expectedContentType);
            stream.Dispose();
        }
        #endregion
    }
}