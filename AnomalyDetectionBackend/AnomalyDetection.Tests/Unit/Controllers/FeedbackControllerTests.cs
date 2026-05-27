using AnomalyDetection.Api.Controllers;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Moq;
using System.Security.Claims;

namespace AnomalyDetection.Tests.Unit.Controllers
{
    public class FeedbackControllerTests
    {
        #region Setup
        private readonly Mock<IFeedbackService> _mockFeedbackService;
        private readonly Mock<ILogger<FeedbackController>> _mockLogger;
        private readonly FeedbackController _controller;

        public FeedbackControllerTests()
        {
            _mockFeedbackService = new Mock<IFeedbackService>();
            _mockLogger = new Mock<ILogger<FeedbackController>>();
            _controller = new FeedbackController(_mockFeedbackService.Object, _mockLogger.Object);

            SetupAuthenticatedUser("1", "User");
        }

        private void SetupAuthenticatedUser(string userId, string role = "User")
        {
            var claims = new List<Claim>
            {
                new Claim("id", userId),
                new Claim(ClaimTypes.Role, role)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var principal = new ClaimsPrincipal(identity);

            _controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };
        }

        private static IFormFile CreateMockFormFile(string fileName = "test.png", long length = 100)
        {
            var mockFile = new Mock<IFormFile>();
            mockFile.Setup(f => f.FileName).Returns(fileName);
            mockFile.Setup(f => f.Length).Returns(length);
            return mockFile.Object;
        }
        #endregion

        #region SubmitFeedback Tests
        [Fact]
        public async Task SubmitFeedback_ReturnsOk_WhenRequestIsValid()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockFeedbackService
                .Setup(s => s.SaveFeedbackImageAsync("bottle", true, mockFile))
                .ReturnsAsync("/FeedbackData/bottle/anomaly/test.png");

            var request = new FeedbackRequest
            {
                Category = "bottle",
                IsActuallyAnomaly = true,
                Image = mockFile
            };

            // Act
            var result = await _controller.SubmitFeedback(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public async Task SubmitFeedback_ReturnsBadRequest_WhenCategoryIsEmpty()
        {
            // Arrange
            var request = new FeedbackRequest
            {
                Category = "",
                IsActuallyAnomaly = true,
                Image = CreateMockFormFile()
            };

            // Act
            var result = await _controller.SubmitFeedback(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Category is required.");
        }

        [Fact]
        public async Task SubmitFeedback_ReturnsBadRequest_WhenCategoryIsWhitespace()
        {
            // Arrange
            var request = new FeedbackRequest
            {
                Category = "   ",
                IsActuallyAnomaly = true,
                Image = CreateMockFormFile()
            };

            // Act
            var result = await _controller.SubmitFeedback(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Category is required.");
        }

        [Fact]
        public async Task SubmitFeedback_ReturnsBadRequest_WhenImageIsNull()
        {
            // Arrange
            var request = new FeedbackRequest
            {
                Category = "bottle",
                IsActuallyAnomaly = true,
                Image = null
            };

            // Act
            var result = await _controller.SubmitFeedback(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Image file is required.");
        }

        [Fact]
        public async Task SubmitFeedback_ReturnsBadRequest_WhenImageIsEmpty()
        {
            // Arrange
            var request = new FeedbackRequest
            {
                Category = "bottle",
                IsActuallyAnomaly = true,
                Image = CreateMockFormFile(length: 0)
            };

            // Act
            var result = await _controller.SubmitFeedback(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Image file is required.");
        }

        [Fact]
        public async Task SubmitFeedback_DoesNotCallService_WhenCategoryIsMissing()
        {
            // Arrange
            var request = new FeedbackRequest
            {
                Category = "",
                IsActuallyAnomaly = true,
                Image = CreateMockFormFile()
            };

            // Act
            await _controller.SubmitFeedback(request);

            // Assert
            _mockFeedbackService.Verify(s => s.SaveFeedbackImageAsync(
                It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<IFormFile>()
            ), Times.Never);
        }

        [Fact]
        public async Task SubmitFeedback_CallsService_WithCorrectParameters()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockFeedbackService
                .Setup(s => s.SaveFeedbackImageAsync("capsule", false, mockFile))
                .ReturnsAsync("/FeedbackData/capsule/good/test.png");

            var request = new FeedbackRequest
            {
                Category = "capsule",
                IsActuallyAnomaly = false,
                Image = mockFile
            };

            // Act
            await _controller.SubmitFeedback(request);

            // Assert
            _mockFeedbackService.Verify(s => s.SaveFeedbackImageAsync("capsule", false, mockFile), Times.Once);
        }

        [Fact]
        public async Task SubmitFeedback_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockFeedbackService
                .Setup(s => s.SaveFeedbackImageAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<IFormFile>()))
                .ThrowsAsync(new Exception("Disk error"));

            var request = new FeedbackRequest
            {
                Category = "bottle",
                IsActuallyAnomaly = true,
                Image = mockFile
            };

            // Act
            var result = await _controller.SubmitFeedback(request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region GetFeedbackSummary Tests
        [Fact]
        public void GetFeedbackSummary_ReturnsOk_WithSummary()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            var summary = new List<object>
            {
                new { Category = "bottle",  AnomalyCount = 5, GoodCount = 10 },
                new { Category = "capsule", AnomalyCount = 2, GoodCount = 8  }
            };
            _mockFeedbackService.Setup(s => s.GetFeedbackSummary()).Returns(summary);

            // Act
            var result = _controller.GetFeedbackSummary();

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(summary);
        }

        [Fact]
        public void GetFeedbackSummary_ReturnsOk_WithEmptyList()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockFeedbackService.Setup(s => s.GetFeedbackSummary()).Returns(new List<object>());

            // Act
            var result = _controller.GetFeedbackSummary();

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void GetFeedbackSummary_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockFeedbackService.Setup(s => s.GetFeedbackSummary())
                                .Throws(new Exception("Unexpected error"));

            // Act
            var result = _controller.GetFeedbackSummary();

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region GetFeedbackImageList Tests
        [Fact]
        public void GetFeedbackImageList_ReturnsOk_WithFileNames()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            var files = new List<string> { "img1.png", "img2.png" };
            _mockFeedbackService.Setup(s => s.GetFeedbackImageNames("bottle", "anomaly")).Returns(files);

            // Act
            var result = _controller.GetFeedbackImageList("bottle", "anomaly");

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(files);
        }

        [Fact]
        public void GetFeedbackImageList_ReturnsBadRequest_WhenLabelIsInvalid()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");

            // Act
            var result = _controller.GetFeedbackImageList("bottle", "invalid_label");

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Theory]
        [InlineData("anomaly")]
        [InlineData("good")]
        public void GetFeedbackImageList_ReturnsOk_ForValidLabels(string label)
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockFeedbackService.Setup(s => s.GetFeedbackImageNames("bottle", label))
                                .Returns(new List<string>());

            // Act
            var result = _controller.GetFeedbackImageList("bottle", label);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void GetFeedbackImageList_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockFeedbackService.Setup(s => s.GetFeedbackImageNames(It.IsAny<string>(), It.IsAny<string>()))
                                .Throws(new Exception("Unexpected error"));

            // Act
            var result = _controller.GetFeedbackImageList("bottle", "anomaly");

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region GetFeedbackImage Tests
        [Fact]
        public void GetFeedbackImage_ReturnsFile_WhenImageExists()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            Stream fakeStream = new MemoryStream(new byte[] { 1, 2, 3 });
            _mockFeedbackService
                .Setup(s => s.GetFeedbackImageStream("bottle", "anomaly", "img1.png"))
                .Returns((fakeStream, "image/png"));

            // Act
            var result = _controller.GetFeedbackImage("bottle", "anomaly", "img1.png");

            // Assert
            result.Should().BeOfType<FileStreamResult>()
                  .Which.ContentType.Should().Be("image/png");
        }

        [Fact]
        public void GetFeedbackImage_ReturnsBadRequest_WhenLabelIsInvalid()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");

            // Act
            var result = _controller.GetFeedbackImage("bottle", "wrong_label", "img1.png");

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void GetFeedbackImage_ReturnsBadRequest_WhenFilenameContainsPathTraversal()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");

            // Act
            var result = _controller.GetFeedbackImage("bottle", "anomaly", "../secrets.txt");

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void GetFeedbackImage_ReturnsNotFound_WhenFileDoesNotExist()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockFeedbackService
                .Setup(s => s.GetFeedbackImageStream("bottle", "anomaly", "missing.png"))
                .Throws(new FileNotFoundException());

            // Act
            var result = _controller.GetFeedbackImage("bottle", "anomaly", "missing.png");

            // Assert
            result.Should().BeOfType<NotFoundObjectResult>();
        }

        [Fact]
        public void GetFeedbackImage_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockFeedbackService
                .Setup(s => s.GetFeedbackImageStream(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                .Throws(new Exception("Unexpected error"));

            // Act
            var result = _controller.GetFeedbackImage("bottle", "anomaly", "img1.png");

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion
    }
}