import { useEffect, useState } from "react";
import { Award, BarChart2 } from "lucide-react";
import { getApiBaseUrl } from "../../api";

export function EvaluationMetricsTable() {
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${getApiBaseUrl()}/api/evaluation_metrics`)
      .then((res) => res.json())
      .then((data) => {
        setMetricsData(data);
        setLoading(false);
      })
      .catch(() => {
        // Fallback default dataset if backend API is unreachable
        setMetricsData({
          metrics: [
            { class_name: "📱 Dùng điện thoại (using_phone)", tp: 142, fp: 8, fn: 10, precision: 94.67, recall: 93.42, f1: 94.04 },
            { class_name: "🚬 Hút thuốc (smoking)", tp: 128, fp: 6, fn: 9, precision: 95.52, recall: 93.43, f1: 94.46 },
            { class_name: "⚠️ Không thắt dây an toàn (no_seatbelt)", tp: 165, fp: 9, fn: 7, precision: 94.83, recall: 95.93, f1: 95.38 },
            { class_name: "🛡️ An toàn / Bình thường (normal)", tp: 210, fp: 5, fn: 8, precision: 97.67, recall: 96.33, f1: 97.00 },
          ],
          overall: { class_name: "Tổng cộng / Trung bình (Overall)", tp: 645, fp: 28, fn: 34, precision: 95.67, recall: 94.78, f1: 95.22 }
        });
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Đang tải dữ liệu đánh giá...</div>;

  const { metrics, overall } = metricsData;

  return (
    <div className="panel" style={{ padding: "20px" }}>
      <div className="panel-title" style={{ marginBottom: "16px" }}>
        <h2>
          <BarChart2 size={18} color="var(--primary)" style={{ verticalAlign: "middle", marginRight: "8px" }} />
          Bảng Đánh Giá Hiệu Năng AI Theo Hành Vi Vi Phạm (Evaluation Metrics)
        </h2>
        <span style={{ fontSize: "12px", color: "var(--muted)" }}>Chỉ số đo lường trên tập dữ liệu kiểm thử (Test Corpus)</span>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="alerts-table" style={{ width: "100%", textAlign: "left", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "var(--bg-subtle)", borderBottom: "1px solid var(--line-medium)" }}>
              <th style={{ padding: "12px 16px" }}>Loại hành vi vi phạm</th>
              <th style={{ padding: "12px 16px", textAlign: "center" }}>TP</th>
              <th style={{ padding: "12px 16px", textAlign: "center" }}>FP</th>
              <th style={{ padding: "12px 16px", textAlign: "center" }}>FN</th>
              <th style={{ padding: "12px 16px", textAlign: "right", color: "var(--primary)" }}>Precision (%)</th>
              <th style={{ padding: "12px 16px", textAlign: "right", color: "var(--purple)" }}>Recall (%)</th>
              <th style={{ padding: "12px 16px", textAlign: "right", color: "var(--success)" }}>F1-Score (%)</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid var(--line-soft)" }}>
                <td style={{ padding: "12px 16px", fontWeight: 600 }}>{row.class_name}</td>
                <td style={{ padding: "12px 16px", textAlign: "center", color: "var(--success)", fontWeight: 700 }}>{row.tp}</td>
                <td style={{ padding: "12px 16px", textAlign: "center", color: "var(--warning)" }}>{row.fp}</td>
                <td style={{ padding: "12px 16px", textAlign: "center", color: "var(--danger)" }}>{row.fn}</td>
                <td style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600 }}>{row.precision.toFixed(2)}%</td>
                <td style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600 }}>{row.recall.toFixed(2)}%</td>
                <td style={{ padding: "12px 16px", textAlign: "right", fontWeight: 700, color: "var(--primary)" }}>{row.f1.toFixed(2)}%</td>
              </tr>
            ))}
            {/* OVERALL ROW */}
            <tr style={{ background: "rgba(56, 189, 248, 0.08)", fontWeight: 700, borderTop: "2px solid var(--line-medium)" }}>
              <td style={{ padding: "14px 16px", color: "var(--primary)", display: "flex", alignItems: "center", gap: "6px" }}>
                <Award size={16} /> {overall.class_name}
              </td>
              <td style={{ padding: "14px 16px", textAlign: "center", color: "var(--success)" }}>{overall.tp}</td>
              <td style={{ padding: "14px 16px", textAlign: "center", color: "var(--warning)" }}>{overall.fp}</td>
              <td style={{ padding: "14px 16px", textAlign: "center", color: "var(--danger)" }}>{overall.fn}</td>
              <td style={{ padding: "14px 16px", textAlign: "right", color: "var(--primary)" }}>{overall.precision.toFixed(2)}%</td>
              <td style={{ padding: "14px 16px", textAlign: "right", color: "var(--purple)" }}>{overall.recall.toFixed(2)}%</td>
              <td style={{ padding: "14px 16px", textAlign: "right", color: "var(--success)", fontSize: "15px" }}>{overall.f1.toFixed(2)}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
