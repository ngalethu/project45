import { useState, useRef, useEffect } from "react";
import { Upload, Camera, Image as ImageIcon, Play, CheckCircle2, AlertTriangle, RefreshCw, Sparkles, Cpu, X, Maximize2, Activity } from "lucide-react";
import { getApiBaseUrl } from "../api";
import { EvaluationMetricsTable } from "../components/common/EvaluationMetricsTable.jsx";

export function TestPage() {
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [detectionError, setDetectionError] = useState(null);
  const [modalMedia, setModalMedia] = useState(null);

  // Webcam state
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isWebcamAiActive, setIsWebcamAiActive] = useState(false);
  const [webcamAiResult, setWebcamAiResult] = useState(null);
  const [webcamAiLogs, setWebcamAiLogs] = useState([]);
  const [webcamFrameCount, setWebcamFrameCount] = useState(0);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const isProcessingRef = useRef(false);
  const webcamIntervalRef = useRef(null);

  useEffect(() => {
    return () => {
      if (webcamIntervalRef.current) {
        clearInterval(webcamIntervalRef.current);
        webcamIntervalRef.current = null;
      }
      isProcessingRef.current = false;
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
    };
  }, []);

  const isImageFile = (f) => {
    if (!f) return false;
    return f.type.startsWith("image/") || /\.(jpg|jpeg|png|webp|bmp)$/i.test(f.name);
  };

  const handleSelectFile = (selected) => {
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setDetectionResult(null);
      setDetectionError(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleClearFile = (e) => {
    if (e) e.stopPropagation();
    setFile(null);
    setPreviewUrl(null);
    setDetectionResult(null);
    setDetectionError(null);
  };

  const handleAutoDetect = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsDetecting(true);
    setDetectionResult(null);
    setDetectionError(null);

    try {
      const formData = new FormData();
      formData.append("media_file", file);
      formData.append("notes", "Tự động nhận diện AI qua Web UI");

      const res = await fetch(`${getApiBaseUrl()}/api/auto_detect_media`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Lỗi nhận diện AI" }));
        throw new Error(err.detail || "Không thể nhận diện AI trên file này");
      }

      const data = await res.json();
      setDetectionResult(data);
    } catch (err) {
      if (err.message === "Failed to fetch") {
        setDetectionError("Không thể kết nối Backend Server (http://127.0.0.1:8000). Đảm bảo backend đã khởi chạy.");
      } else {
        setDetectionError(err.message);
      }
    } finally {
      setIsDetecting(false);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsWebcamActive(true);
      return true;
    } catch (err) {
      alert("Không thể truy cập Webcam: " + err.message);
      return false;
    }
  };

  const stopWebcam = () => {
    stopWebcamAi();
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
  };

  const startWebcamAi = async () => {
    if (!isWebcamActive) {
      const ok = await startWebcam();
      if (!ok) return;
    }

    setIsWebcamAiActive(true);
    isProcessingRef.current = false;

    webcamIntervalRef.current = setInterval(async () => {
      if (isProcessingRef.current || !videoRef.current || videoRef.current.paused || videoRef.current.ended) {
        return;
      }

      const v = videoRef.current;
      if (!v.videoWidth || !v.videoHeight) return;

      isProcessingRef.current = true;

      try {
        const canvas = document.createElement("canvas");
        canvas.width = v.videoWidth;
        canvas.height = v.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
          if (!blob) {
            isProcessingRef.current = false;
            return;
          }

          const formData = new FormData();
          formData.append("media_file", blob, "webcam_frame.jpg");
          formData.append("notes", "Live Web UI Camera Stream AI");

          try {
            const res = await fetch(`${getApiBaseUrl()}/api/auto_detect_media`, {
              method: "POST",
              body: formData,
            });

            if (res.ok) {
              const data = await res.json();
              setWebcamAiResult(data);
              setWebcamFrameCount((prev) => prev + 1);
              setWebcamAiLogs((prev) => [data, ...prev.slice(0, 4)]);
            }
          } catch (err) {
            console.error("Lỗi Live Webcam AI:", err);
          } finally {
            isProcessingRef.current = false;
          }
        }, "image/jpeg", 0.85);
      } catch (e) {
        console.error("Lỗi capture canvas webcam:", e);
        isProcessingRef.current = false;
      }
    }, 400); // 2.5 FPS for fast live response
  };

  const stopWebcamAi = () => {
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
    setIsWebcamAiActive(false);
    isProcessingRef.current = false;
  };

  const getEventBadge = (type) => {
    switch (type) {
      case "using_phone":
        return <span className="badge danger">📱 Dùng điện thoại (using_phone)</span>;
      case "no_seatbelt":
        return <span className="badge danger">⚠️ Không thắt dây an toàn (no_seatbelt)</span>;
      default:
        return <span className="badge success">🛡️ An toàn / Bình thường (normal)</span>;
    }
  };

  return (
    <div className="page">
      <section className="page-header">
        <div>
          <h1>Tự Động Nhận Diện AI & Đánh Giá Hiệu Năng</h1>
          <p>
            Tải lên file Ảnh/Video hoặc bật Live Webcam để Model YOLO11 + MediaPipe tự động phân loại hành vi vi phạm thời gian thực.
          </p>
        </div>
      </section>

      {/* Tabs selector */}
      <div style={{ display: "flex", gap: "12px", marginBottom: "20px", flexWrap: "wrap" }}>
        <button
          className={`btn ${activeTab === "upload" ? "primary-btn" : "secondary-btn"}`}
          onClick={() => setActiveTab("upload")}
        >
          <Sparkles size={16} /> Tự Động Nhận Diện AI (Ảnh / Video)
        </button>
        <button
          className={`btn ${activeTab === "webcam" ? "primary-btn" : "secondary-btn"}`}
          onClick={() => setActiveTab("webcam")}
        >
          <Camera size={16} /> Live Webcam AI Detection
        </button>
        <button
          className={`btn ${activeTab === "metrics" ? "primary-btn" : "secondary-btn"}`}
          onClick={() => setActiveTab("metrics")}
        >
          <Cpu size={16} /> Bảng Đánh Giá Hiệu Năng (TP, FP, FN, Precision, Recall, F1)
        </button>
        <button
          className={`btn ${activeTab === "cli" ? "primary-btn" : "secondary-btn"}`}
          onClick={() => setActiveTab("cli")}
        >
          <Play size={16} /> Hướng Dẫn Run Script CLI
        </button>
      </div>

      {/* TAB 1: UPLOAD & AUTO AI DETECT */}
      {activeTab === "upload" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
            <div className="panel" style={{ padding: "24px" }}>
              <h2 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                <Upload size={18} color="var(--primary)" /> Tải File Ảnh Hoặc Video (AI Tự Phân Loại)
              </h2>

              <form onSubmit={handleAutoDetect} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div
                  style={{
                    border: `2px dashed ${isDragging ? "var(--primary)" : "var(--line-medium)"}`,
                    borderRadius: "var(--radius-md)",
                    padding: "32px 16px",
                    textAlign: "center",
                    background: isDragging ? "rgba(59, 130, 246, 0.08)" : "var(--bg-subtle)",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    position: "relative",
                  }}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById("file-input-auto").click()}
                >
                  <input
                    id="file-input-auto"
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileChange}
                    style={{ display: "none" }}
                  />
                  {previewUrl ? (
                    <div style={{ position: "relative" }}>
                      <button
                        type="button"
                        onClick={handleClearFile}
                        title="Đổi file khác"
                        style={{
                          position: "absolute",
                          top: "4px",
                          right: "4px",
                          background: "rgba(0,0,0,0.6)",
                          color: "#fff",
                          border: "none",
                          borderRadius: "50%",
                          width: "28px",
                          height: "28px",
                          display: "grid",
                          placeItems: "center",
                          cursor: "pointer",
                          zIndex: 10,
                        }}
                      >
                        <X size={16} />
                      </button>

                      {isImageFile(file) ? (
                        <img src={previewUrl} alt="Preview" style={{ maxHeight: "220px", borderRadius: "8px", maxWidth: "100%", objectFit: "contain" }} />
                      ) : (
                        <video src={previewUrl} controls style={{ maxHeight: "220px", borderRadius: "8px", width: "100%" }} />
                      )}
                      <p style={{ marginTop: "8px", fontSize: "12px", color: "var(--muted)", fontWeight: 500 }}>
                        {file?.name} ({(file?.size / (1024 * 1024)).toFixed(2)} MB)
                      </p>
                    </div>
                  ) : (
                    <div>
                      <Upload size={40} color={isDragging ? "var(--primary)" : "var(--muted)"} style={{ marginBottom: "12px", transition: "transform 0.2s" }} />
                      <p style={{ fontWeight: 600, margin: "0 0 4px", fontSize: "14px" }}>
                        {isDragging ? "Thả file vào đây để tải lên" : "Nhấp để chọn hoặc kéo thả file Ảnh / Video vào đây"}
                      </p>
                      <p style={{ fontSize: "12px", color: "var(--muted)", margin: 0 }}>
                        Model YOLO11 sẽ tự động phân tích, phát hiện vi phạm và vẽ bboxes
                      </p>
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  className="primary-btn"
                  disabled={!file || isDetecting}
                  style={{ justifyContent: "center", padding: "14px", fontSize: "14px" }}
                >
                  {isDetecting ? <RefreshCw className="spin" size={18} /> : <Sparkles size={18} />}
                  {isDetecting ? "AI Đang Phân Tích & Phân Loại..." : "Chạy AI Tự Động Nhận Diện Hành Vi"}
                </button>
              </form>

              {detectionError && (
                <div className="error-banner" style={{ marginTop: "16px" }}>
                  <AlertTriangle size={16} />
                  <span>{detectionError}</span>
                </div>
              )}
            </div>

            {/* AI Result panel */}
            <div className="panel" style={{ padding: "24px" }}>
              <h2 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                <CheckCircle2 size={18} color="var(--success)" /> Kết Quả Tự Động Phân Loại Của AI
              </h2>

              {isDetecting ? (
                <div style={{ textAlign: "center", padding: "60px 20px" }}>
                  <RefreshCw size={44} className="spin" style={{ color: "var(--primary)", marginBottom: "16px" }} />
                  <p style={{ fontWeight: 600, fontSize: "15px", margin: "0 0 6px" }}>Đang Chạy Mô Hình YOLO11 & MediaPipe...</p>
                  <small style={{ color: "var(--muted)" }}>Hệ thống đang quét từng frame hình, trích xuất đặc trưng tư thế và khoanh vùng vị trí vi phạm.</small>
                </div>
              ) : detectionResult ? (
                <div style={{ background: "var(--bg-subtle)", padding: "18px", borderRadius: "var(--radius-md)" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
                    <span style={{ fontSize: "13px", fontWeight: 700 }}>Hành vi nhận diện được:</span>
                    {getEventBadge(detectionResult.detection.event_type)}
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", fontSize: "13px", marginBottom: "14px" }}>
                    <div><strong>Độ tin cậy (Confidence):</strong> <span style={{ color: "var(--primary)", fontWeight: 700 }}>{(detectionResult.detection.confidence * 100).toFixed(1)}%</span></div>
                    <div><strong>Mã Cảnh Báo DB:</strong> #{detectionResult.alert.id}</div>
                    {detectionResult.detection.detections_count !== undefined && (
                      <div><strong>Đối tượng phát hiện:</strong> {detectionResult.detection.detections_count} vị trí</div>
                    )}
                    {detectionResult.detection.total_frames !== undefined && (
                      <div><strong>Tổng số khung hình:</strong> {detectionResult.detection.total_frames} frames</div>
                    )}
                  </div>

                  {/* Frame bounding box preview */}
                  {detectionResult.detection.frame_url && (
                    <div style={{ marginTop: "14px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                        <span style={{ fontSize: "12px", fontWeight: 600 }}>Ảnh Khoanh Vùng AI (Annotated Frame):</span>
                        <button
                          className="text-btn"
                          style={{ fontSize: "11px", display: "flex", alignItems: "center", gap: "4px", padding: "2px 6px" }}
                          onClick={() => setModalMedia({ type: "image", url: getApiBaseUrl() + detectionResult.detection.frame_url })}
                        >
                          <Maximize2 size={12} /> Xem ảnh lớn
                        </button>
                      </div>
                      <img
                        src={getApiBaseUrl() + detectionResult.detection.frame_url}
                        alt="AI Detected Frame"
                        style={{ width: "100%", borderRadius: "8px", border: "1px solid var(--line-medium)", cursor: "pointer" }}
                        onClick={() => setModalMedia({ type: "image", url: getApiBaseUrl() + detectionResult.detection.frame_url })}
                      />
                    </div>
                  )}

                  {/* Video clip preview */}
                  {detectionResult.detection.clip_url && (
                    <div style={{ marginTop: "14px" }}>
                      <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "6px" }}>Video Clip Xử Lý AI:</div>
                      <video src={getApiBaseUrl() + detectionResult.detection.clip_url} controls style={{ width: "100%", borderRadius: "8px", border: "1px solid var(--line-medium)" }} />
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: "50px 20px", color: "var(--muted)" }}>
                  <ImageIcon size={48} style={{ opacity: 0.3, marginBottom: "12px" }} />
                  <p style={{ margin: 0, fontWeight: 500 }}>Vui lòng tải lên file bên trái và bấm <strong>Chạy AI Tự Động Nhận Diện</strong>.</p>
                  <small style={{ display: "block", marginTop: "6px", color: "var(--muted)" }}>Model sẽ tự động phát hiện hành vi dùng điện thoại, hút thuốc hoặc không thắt dây an toàn.</small>
                </div>
              )}
            </div>
          </div>

          {/* Render Evaluation Metrics Table below upload panel */}
          <EvaluationMetricsTable />
        </div>
      )}

      {/* TAB 2: LIVE WEBCAM AI DETECTION */}
      {activeTab === "webcam" && (
        <div className="panel" style={{ padding: "24px" }}>
          <h2 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <Camera size={18} color="var(--primary)" /> Nhận Diện AI Trực Tiếp Qua Browser Webcam
          </h2>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            {/* Left Col: Stream & Control */}
            <div>
              <p style={{ fontSize: "13px", color: "var(--muted)", marginBottom: "16px" }}>
                Bật Camera và kích hoạt <strong>Live AI Detection</strong> để trình duyệt tự động trích xuất khung hình gửi về YOLO11 + MediaPipe xử lý liên tục.
              </p>

              <div style={{ display: "flex", gap: "10px", marginBottom: "20px", flexWrap: "wrap" }}>
                {!isWebcamActive ? (
                  <button className="primary-btn" onClick={startWebcam}>
                    <Camera size={16} /> Bật Camera Trình Duyệt
                  </button>
                ) : (
                  <button className="secondary-btn" onClick={stopWebcam}>
                    Tắt Camera
                  </button>
                )}

                {!isWebcamAiActive ? (
                  <button
                    className="primary-btn"
                    onClick={startWebcamAi}
                    style={{ background: "linear-gradient(135deg, #10b981 0%, #059669 100%)", borderColor: "#059669" }}
                  >
                    <Sparkles size={16} /> Bật AI Live Detection
                  </button>
                ) : (
                  <button className="danger-btn" onClick={stopWebcamAi}>
                    <X size={16} /> Dừng Live AI Detection
                  </button>
                )}
              </div>

              {/* Raw Video Container */}
              <div style={{ background: "#000", borderRadius: "12px", overflow: "hidden", minHeight: "280px", display: "grid", placeItems: "center", position: "relative" }}>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{ width: "100%", maxHeight: "320px", display: isWebcamActive ? "block" : "none" }}
                />
                {!isWebcamActive && (
                  <div style={{ color: "#94a3b8", textAlign: "center" }}>
                    <Camera size={40} style={{ opacity: 0.4, marginBottom: "8px" }} />
                    <p style={{ margin: 0, fontSize: "13px" }}>Webcam đang tắt</p>
                  </div>
                )}

                {isWebcamAiActive && (
                  <div style={{
                    position: "absolute",
                    top: "10px",
                    left: "10px",
                    background: "rgba(16, 185, 129, 0.9)",
                    color: "#fff",
                    padding: "4px 10px",
                    borderRadius: "20px",
                    fontSize: "11px",
                    fontWeight: 700,
                    display: "flex",
                    alignItems: "center",
                    gap: "6px"
                  }}>
                    <Activity size={12} className="spin" /> AI STREAMING ACTIVE ({webcamFrameCount} frames)
                  </div>
                )}
              </div>
            </div>

            {/* Right Col: Live AI Detection Result */}
            <div style={{ background: "var(--bg-subtle)", padding: "20px", borderRadius: "var(--radius-md)" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "14px", display: "flex", alignItems: "center", gap: "8px" }}>
                <CheckCircle2 size={16} color="var(--success)" /> Live AI Bounding Boxes Feed
              </h3>

              {webcamAiResult ? (
                <div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                    <span style={{ fontSize: "12px", fontWeight: 700 }}>Hành Vi Hiện Tại:</span>
                    {getEventBadge(webcamAiResult.detection.event_type)}
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", fontSize: "12px", marginBottom: "14px" }}>
                    <div><strong>Độ Tin Cậy:</strong> <span style={{ color: "var(--primary)", fontWeight: 700 }}>{(webcamAiResult.detection.confidence * 100).toFixed(1)}%</span></div>
                    <div><strong>Số Vị Trí Bbox:</strong> {webcamAiResult.detection.detections_count}</div>
                  </div>

                  {webcamAiResult.detection.frame_url && (
                    <div>
                      <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "6px" }}>Frame Khoanh Vùng AI Mới Nhất:</div>
                      <img
                        src={getApiBaseUrl() + webcamAiResult.detection.frame_url}
                        alt="Live AI Frame"
                        style={{ width: "100%", maxHeight: "220px", objectFit: "contain", borderRadius: "8px", border: "1px solid var(--line-medium)" }}
                      />
                    </div>
                  )}

                  {/* History Log */}
                  {webcamAiLogs.length > 0 && (
                    <div style={{ marginTop: "14px" }}>
                      <div style={{ fontSize: "11px", fontWeight: 700, marginBottom: "6px", color: "var(--muted)" }}>Lịch Sử Nhận Diện Gần Đây:</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        {webcamAiLogs.map((log, idx) => (
                          <div key={idx} style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", background: "#fff", padding: "4px 8px", borderRadius: "4px", border: "1px solid var(--line)" }}>
                            <span>Alert #{log.alert.id}</span>
                            <span>{log.detection.event_type}</span>
                            <span style={{ color: "var(--primary)", fontWeight: 600 }}>{(log.detection.confidence * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--muted)" }}>
                  <Sparkles size={40} style={{ opacity: 0.3, marginBottom: "12px" }} />
                  <p style={{ margin: 0, fontSize: "13px", fontWeight: 500 }}>
                    Nhấn nút <strong>Bật AI Live Detection</strong> bên trái để khởi chạy nhận diện tư thế & vi phạm từ Webcam.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: METRICS TABLE */}
      {activeTab === "metrics" && (
        <EvaluationMetricsTable />
      )}

      {/* TAB 4: CLI */}
      {activeTab === "cli" && (
        <div className="panel" style={{ padding: "24px" }}>
          <h2 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <Play size={18} color="var(--primary)" /> Các Lệnh Test Dự Án (Command Line Reference)
          </h2>

          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "16px", borderRadius: "var(--radius-md)" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, margin: "0 0 6px" }}>1. Test nhận diện trên Video bất kỳ</h3>
              <pre style={{ background: "#0f172a", color: "#38bdf8", padding: "10px 14px", borderRadius: "6px", fontSize: "12px" }}>
                {`py -3.11 -m scripts.test_video --video "data/sample_videos/test.mp4" --send_to_cloud`}
              </pre>
            </div>

            <div style={{ background: "var(--bg-subtle)", padding: "16px", borderRadius: "var(--radius-md)" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, margin: "0 0 6px" }}>2. Test nhận diện trên Ảnh bất kỳ</h3>
              <pre style={{ background: "#0f172a", color: "#38bdf8", padding: "10px 14px", borderRadius: "6px", fontSize: "12px" }}>
                {`py -3.11 -m scripts.test_image --image "path/to/image.jpg" --save "outputs/result.jpg"`}
              </pre>
            </div>

            <div style={{ background: "var(--bg-subtle)", padding: "16px", borderRadius: "var(--radius-md)" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, margin: "0 0 6px" }}>3. Chạy toàn bộ Benchmark đo FPS</h3>
              <pre style={{ background: "#0f172a", color: "#38bdf8", padding: "10px 14px", borderRadius: "6px", fontSize: "12px" }}>
                {`py -3.11 -m scripts.run_all_benchmarks`}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* MODAL FOR FULL MEDIA VIEW */}
      {modalMedia && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0, 0, 0, 0.85)",
            zIndex: 9999,
            display: "grid",
            placeItems: "center",
            padding: "20px",
          }}
          onClick={() => setModalMedia(null)}
        >
          <div style={{ position: "relative", maxWidth: "90vw", maxHeight: "90vh" }} onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setModalMedia(null)}
              style={{
                position: "absolute",
                top: "-36px",
                right: "0",
                background: "transparent",
                color: "#fff",
                border: "none",
                fontSize: "18px",
                cursor: "pointer",
              }}
            >
              ✕ Đóng
            </button>
            <img
              src={modalMedia.url}
              alt="Annotated Frame Full"
              style={{ maxWidth: "90vw", maxHeight: "85vh", borderRadius: "8px", objectFit: "contain" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
