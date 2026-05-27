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
    public class ModelsControllerTests
    {
        #region Setup
        private readonly Mock<IModelManagerService> _mockModelManager;
        private readonly Mock<ILogger<ModelsController>> _mockLogger;
        private readonly ModelsController _controller;

        public ModelsControllerTests()
        {
            _mockModelManager = new Mock<IModelManagerService>();
            _mockLogger = new Mock<ILogger<ModelsController>>();
            _controller = new ModelsController(_mockModelManager.Object, _mockLogger.Object);

            SetupAuthenticatedUser("1", "Admin");
        }

        private void SetupAuthenticatedUser(string userId, string role = "Admin")
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

        private static IFormFile CreateMockFormFile(string fileName = "model.npz", long length = 100)
        {
            var mockFile = new Mock<IFormFile>();
            var stream = new MemoryStream(new byte[length]);

            mockFile.Setup(f => f.FileName).Returns(fileName);
            mockFile.Setup(f => f.Length).Returns(length);
            mockFile.Setup(f => f.CopyToAsync(It.IsAny<Stream>(), It.IsAny<CancellationToken>()))
                    .Returns(Task.CompletedTask);

            return mockFile.Object;
        }
        #endregion

        #region GetModels Tests
        [Fact]
        public void GetModels_ReturnsOk_WithModelList()
        {
            // Arrange
            var models = new List<ModelInfo>
            {
                new ModelInfo { Category = "bottle",  Threshold = 0.5f },
                new ModelInfo { Category = "capsule", Threshold = 0.6f }
            };
            _mockModelManager.Setup(m => m.GetAvailableModels()).Returns(models);

            // Act
            var result = _controller.GetModels();

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(models);
        }

        [Fact]
        public void GetModels_ReturnsOk_WithEmptyList_WhenNoModelsAvailable()
        {
            // Arrange
            _mockModelManager.Setup(m => m.GetAvailableModels()).Returns(new List<ModelInfo>());

            // Act
            var result = _controller.GetModels();

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.As<List<ModelInfo>>().Should().BeEmpty();
        }

        [Fact]
        public void GetModels_CallsService_Once()
        {
            // Arrange
            _mockModelManager.Setup(m => m.GetAvailableModels()).Returns(new List<ModelInfo>());

            // Act
            _controller.GetModels();

            // Assert
            _mockModelManager.Verify(m => m.GetAvailableModels(), Times.Once);
        }
        #endregion

        #region UploadModel Tests
        [Fact]
        public async Task UploadModel_ReturnsOk_WhenFilesAreValid()
        {
            // Arrange
            var bankFile = CreateMockFormFile("patchcore_memory_bottle.npz");
            var metaFile = CreateMockFormFile("metadata_bottle.json");

            _mockModelManager
                .Setup(m => m.UploadNewModelAsync("bottle", bankFile, metaFile))
                .Returns(Task.CompletedTask);

            var request = new UploadModelRequest
            {
                Category = "bottle",
                BankFile = bankFile,
                JsonMetadata = metaFile
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public async Task UploadModel_ReturnsBadRequest_WhenCategoryIsEmpty()
        {
            // Arrange
            var request = new UploadModelRequest
            {
                Category = "",
                BankFile = CreateMockFormFile("model.npz"),
                JsonMetadata = CreateMockFormFile("meta.json")
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public async Task UploadModel_ReturnsBadRequest_WhenBankFileIsNull()
        {
            // Arrange
            var request = new UploadModelRequest
            {
                Category = "bottle",
                BankFile = null,
                JsonMetadata = CreateMockFormFile("meta.json")
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public async Task UploadModel_ReturnsBadRequest_WhenBankFileHasWrongExtension()
        {
            // Arrange
            var request = new UploadModelRequest
            {
                Category = "bottle",
                BankFile = CreateMockFormFile("model.zip"),
                JsonMetadata = CreateMockFormFile("meta.json")
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public async Task UploadModel_ReturnsBadRequest_WhenMetadataFileIsNull()
        {
            // Arrange
            var request = new UploadModelRequest
            {
                Category = "bottle",
                BankFile = CreateMockFormFile("model.npz"),
                JsonMetadata = null
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public async Task UploadModel_ReturnsBadRequest_WhenMetadataFileHasWrongExtension()
        {
            // Arrange
            var request = new UploadModelRequest
            {
                Category = "bottle",
                BankFile = CreateMockFormFile("model.npz"),
                JsonMetadata = CreateMockFormFile("meta.txt")
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public async Task UploadModel_CallsService_WithCorrectParameters()
        {
            // Arrange
            var bankFile = CreateMockFormFile("model.npz");
            var metaFile = CreateMockFormFile("meta.json");

            _mockModelManager
                .Setup(m => m.UploadNewModelAsync("capsule", bankFile, metaFile))
                .Returns(Task.CompletedTask);

            var request = new UploadModelRequest
            {
                Category = "capsule",
                BankFile = bankFile,
                JsonMetadata = metaFile
            };

            // Act
            await _controller.UploadModel(request);

            // Assert
            _mockModelManager.Verify(m => m.UploadNewModelAsync("capsule", bankFile, metaFile), Times.Once);
        }

        [Fact]
        public async Task UploadModel_DoesNotCallService_WhenValidationFails()
        {
            // Arrange
            var request = new UploadModelRequest
            {
                Category = "",
                BankFile = CreateMockFormFile("model.npz"),
                JsonMetadata = CreateMockFormFile("meta.json")
            };

            // Act
            await _controller.UploadModel(request);

            // Assert
            _mockModelManager.Verify(m => m.UploadNewModelAsync(
                It.IsAny<string>(), It.IsAny<IFormFile>(), It.IsAny<IFormFile>()
            ), Times.Never);
        }

        [Fact]
        public async Task UploadModel_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            var bankFile = CreateMockFormFile("model.npz");
            var metaFile = CreateMockFormFile("meta.json");

            _mockModelManager
                .Setup(m => m.UploadNewModelAsync(It.IsAny<string>(), It.IsAny<IFormFile>(), It.IsAny<IFormFile>()))
                .ThrowsAsync(new Exception("Disk error"));

            var request = new UploadModelRequest
            {
                Category = "bottle",
                BankFile = bankFile,
                JsonMetadata = metaFile
            };

            // Act
            var result = await _controller.UploadModel(request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region DeleteModel Tests
        [Fact]
        public void DeleteModel_ReturnsOk_WhenCategoryExists()
        {
            // Arrange
            _mockModelManager.Setup(m => m.DeleteModel("bottle"));

            // Act
            var result = _controller.DeleteModel("bottle");

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void DeleteModel_ReturnsBadRequest_WhenCategoryIsEmpty()
        {
            // Act
            var result = _controller.DeleteModel("");

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void DeleteModel_ReturnsBadRequest_WhenCategoryIsWhitespace()
        {
            // Act
            var result = _controller.DeleteModel("   ");

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void DeleteModel_ReturnsNotFound_WhenModelFilesDoNotExist()
        {
            // Arrange
            _mockModelManager.Setup(m => m.DeleteModel("nonexistent"))
                             .Throws(new FileNotFoundException("Model not found"));

            // Act
            var result = _controller.DeleteModel("nonexistent");

            // Assert
            result.Should().BeOfType<NotFoundObjectResult>();
        }

        [Fact]
        public void DeleteModel_ReturnsBadRequest_WhenCategoryNameIsInvalid()
        {
            // Arrange
            _mockModelManager.Setup(m => m.DeleteModel("../malicious"))
                             .Throws(new ArgumentException("Invalid category name — path traversal not allowed."));

            // Act
            var result = _controller.DeleteModel("../malicious");

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void DeleteModel_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            _mockModelManager.Setup(m => m.DeleteModel(It.IsAny<string>()))
                             .Throws(new Exception("Unexpected error"));

            // Act
            var result = _controller.DeleteModel("bottle");

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }

        [Fact]
        public void DeleteModel_CallsService_WithCorrectCategory()
        {
            // Arrange
            _mockModelManager.Setup(m => m.DeleteModel("capsule"));

            // Act
            _controller.DeleteModel("capsule");

            // Assert
            _mockModelManager.Verify(m => m.DeleteModel("capsule"), Times.Once);
        }

        [Fact]
        public void DeleteModel_DoesNotCallService_WhenCategoryIsEmpty()
        {
            // Act
            _controller.DeleteModel("");

            // Assert
            _mockModelManager.Verify(m => m.DeleteModel(It.IsAny<string>()), Times.Never);
        }
        #endregion
    }
}