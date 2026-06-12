FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY AnomalyDetectionBackend.slnx ./
COPY AnomalyDetection.Api/AnomalyDetection.Api.csproj AnomalyDetection.Api/
COPY AnomalyDetection.Tests/AnomalyDetection.Tests.csproj AnomalyDetection.Tests/
RUN dotnet restore AnomalyDetection.Api/AnomalyDetection.Api.csproj

COPY . .
RUN dotnet publish AnomalyDetection.Api/AnomalyDetection.Api.csproj \
    -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgdiplus \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/publish .

EXPOSE 5165

ENV ASPNETCORE_URLS=http://+:5165
ENV ASPNETCORE_ENVIRONMENT=Production
ENV LD_LIBRARY_PATH=/app/runtimes/linux-x64/native:$LD_LIBRARY_PATH

ENTRYPOINT ["dotnet", "AnomalyDetection.Api.dll"]