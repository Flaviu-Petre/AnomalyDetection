using System;
using System.Collections.Generic;
using System.Text;

namespace anomaly_detection_app.Models
{
    public class AnomalyResult
    {
        public float Score { get; set; }
        public byte[] HeatmapImageBytes { get; set; }
    }
}
