// src/App.js

import React, { useState, useEffect } from "react";
import { io } from "socket.io-client";
import { Line } from "react-chartjs-2";
import {
  Chart,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip,
  Legend,
} from "chart.js";
import "./index.css";

// Register Chart.js components
Chart.register(LineElement, PointElement, LinearScale, Title, CategoryScale, Tooltip, Legend);

// Point to your public/architecture.png
const ARCH_IMG = process.env.PUBLIC_URL + "/architecture.png";

// Connect to backend
const socket = io("http://localhost:5000");

// Neon graph colors
const graphColors = {
  "5G": "#FF0000",
  "LTE": "#FF00FF",
  "Radar": "#00FF00",
  "JSSS": "#FFA500",
  "All": "#FFFFFF",
};

function SignalChart({ label, dataKey, graphData }) {
  const recent = graphData.slice(-10);
  const data = {
    labels: recent.map((p) => p.time),
    datasets: [
      {
        label,
        data: recent.map((p) => p[dataKey]),
        fill: false,
        backgroundColor: graphColors[label],
        borderColor: graphColors[label],
      },
    ],
  };
  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#FFF" } },
      title: { display: true, text: label, color: "#FFF" },
    },
    scales: {
      x: { ticks: { color: "#FFF" }, grid: { color: "rgba(255,255,255,0.2)" } },
      y: { ticks: { color: "#FFF" }, grid: { color: "rgba(255,255,255,0.2)" } },
    },
  };
  return <Line data={data} options={options} />;
}

function InfoTabs({ onClose }) {
  const [activeTab, setActiveTab] = useState("Overview");
  const tabs = {
    Overview: (
      <>
        <p>
          Our pipeline transforms raw spectrogram PNGs into annotated insights every 2 seconds:
        </p>
        <ol>
          <li><strong>Ingest:</strong> Watch a folder for new spectrograms.</li>
          <li><strong>Detect:</strong> Run a custom‑trained YOLOv5 model.</li>
          <li><strong>Annotate:</strong> Draw neon bounding boxes with OpenCV.</li>
          <li><strong>Quantify:</strong> Compute interference ratios per band.</li>
          <li><strong>Stream:</strong> Push frames & metrics via Socket.IO.</li>
        </ol>
      </>
    ),

    Architecture: (
      <>
        <p>End‑to‑end system architecture:</p>
        <img
          src={ARCH_IMG}
          alt="System Architecture Diagram"
          style={{ maxWidth: "100%", margin: "10px 0", borderRadius: "4px" }}
        />
        <ul>
          <li>• PNG folder + optional Pascal‑VOC XML</li>
          <li>• Flask + Torch Hub for YOLOv5 inference</li>
          <li>• OpenCV for neon box annotation</li>
          <li>• Socket.IO for real‑time streaming</li>
          <li>• MongoDB for time‑series storage</li>
        </ul>
      </>
    ),

    "How It Works": (
      <>
        <p>Detailed flow every 2 seconds:</p>
        <h4>1. Acquire & Parse</h4>
        <p>Load image in Pillow; parse XML if present.</p>
        <h4>2. YOLO & Annotate</h4>
        <p>Infer with YOLOv5; overlay boxes & labels via OpenCV.</p>
        <h4>3. Compute Metrics</h4>
        <p>Calculate box‑area percentage per band and overall.</p>
        <h4>4. Archive & Alert</h4>
        <p>
          If total ≥50%, save to <code>high_interference/</code> and flash a red banner.
        </p>
      </>
    ),

    Features: (
      <>
        <p>Key features:</p>
        <ul>
          <li>🔴 Multi‑band detection: 5G, LTE, Radar, JSSS</li>
          <li>📸 Annotated outputs</li>
          <li>⏱ Live updates every 2s</li>
          <li>📈 Historical graphs persisted in MongoDB</li>
          <li>⚠️ Configurable noise alerts</li>
          <li>🛠️ One‑click Start/Stop/Reset</li>
        </ul>
      </>
    ),

    Technologies: (
      <>
        <p>Tech stack:</p>
        <ul>
          <li>YOLOv5 (Ultralytics)</li>
          <li>Flask & Socket.IO</li>
          <li>OpenCV & Pillow</li>
          <li>React & Chart.js</li>
          <li>MongoDB</li>
        </ul>
      </>
    ),
  };

  return (
    <div
      className="info-tabs"
      style={{
        position: "fixed",
        top: "10%",
        left: "10%",
        right: "10%",
        bottom: "10%",
        backgroundColor: "#222",
        color: "#FFF",
        padding: "20px",
        borderRadius: "10px",
        boxShadow: "0 0 10px rgba(255,255,255,0.5)",
        zIndex: 1000,
        overflowY: "auto",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h2>Data Mine RTX – Signal Classification Project</h2>
        <button
          onClick={onClose}
          className="close-btn"
          style={{
            background: "#FFF",
            color: "#000",
            border: "none",
            borderRadius: "5px",
            padding: "5px 10px",
            cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>

      <div style={{ borderBottom: "1px solid #555", margin: "10px 0" }}>
        {Object.keys(tabs).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              background: activeTab === tab ? "#FFF" : "transparent",
              color: activeTab === tab ? "#000" : "#FFF",
              border: "none",
              marginRight: "10px",
              padding: "5px 10px",
              cursor: "pointer",
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      <div style={{ padding: "10px", lineHeight: 1.5 }}>
        {tabs[activeTab]}
      </div>
    </div>
  );
}

export default function App() {
  const [spectrogram, setSpectrogram] = useState(null);
  const [graphData, setGraphData] = useState([]);
  const [statusMsg, setStatusMsg] = useState("");
  const [timeStamp, setTimeStamp] = useState(null);
  const [warning, setWarning] = useState("");
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    socket.on("new_detection", (data) => {
      if (data.image) setSpectrogram(`data:image/jpeg;base64,${data.image}`);
      if (data.graphData) {
        setGraphData(data.graphData);
        const pct = data.graphData.slice(-1)[0]?.All * 100;
        setWarning(
          pct >= 50
            ? "Significant interference. Immediate attention required!"
            : pct >= 25
            ? "Moderate interference detected."
            : ""
        );
      }
      if (data.time) setTimeStamp(data.time);
    });
    return () => socket.off("new_detection");
  }, []);

  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `
      @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const bannerStyle = () => {
    if (warning.startsWith("Significant"))
      return {
        margin: "10px",
        padding: "10px",
        backgroundColor: "#F00",
        color: "#FFF",
        fontWeight: "bold",
        textAlign: "center",
        animation: "pulse 2s infinite",
      };
    if (warning.startsWith("Moderate"))
      return {
        margin: "10px",
        padding: "10px",
        backgroundColor: "#FFA500",
        color: "#000",
        fontWeight: "bold",
        textAlign: "center",
      };
    return {};
  };

  const sendCommand = async (cmd) => {
    try {
      const res = await fetch(`http://localhost:5000/${cmd}`, { method: "POST" });
      const body = await res.json();
      setStatusMsg(body.message);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ backgroundColor: "#000", color: "#FFF", minHeight: "100vh", margin: 0 }}>
      <h1 style={{ textAlign: "center", padding: "20px 0", margin: 0 }}>
        Data Mine RTX – Signal Classification Project
      </h1>

      <div style={{ display: "flex", justifyContent: "center", gap: "10px", padding: "10px" }}>
        <button onClick={() => sendCommand("start")} style={{ padding: "10px 20px", borderRadius: "5px" }}>
          Start
        </button>
        <button
          onClick={() => sendCommand("stop")}
          style={{ padding: "10px 20px", backgroundColor: "#F00", color: "#FFF", borderRadius: "5px" }}
        >
          Stop
        </button>
        <button onClick={() => sendCommand("reset")} style={{ padding: "10px 20px", borderRadius: "5px" }}>
          Reset
        </button>
        <button
          onClick={() => setShowInfo(true)}
          className="info-button"
          style={{ padding: "10px 20px", backgroundColor: "#FFD700", borderRadius: "5px" }}
        >
          Info
        </button>
      </div>

      {showInfo && <InfoTabs onClose={() => setShowInfo(false)} />}

      <div style={{ display: "flex", padding: "10px", gap: "20px" }}>
        <div style={{ flex: 1 }}>
          {spectrogram ? (
            <img src={spectrogram} alt="Spectrogram" style={{ width: "100%" }} />
          ) : (
            <p style={{ textAlign: "center" }}>Waiting for spectrogram…</p>
          )}
          {timeStamp && <p style={{ textAlign: "center" }}>Time: {timeStamp}</p>}
          {warning && (
            <div style={bannerStyle()}>
              <strong>⚠️ {warning}</strong>
            </div>
          )}
        </div>

        <div style={{ flex: 1 }}>
          <SignalChart label="5G" dataKey="5G" graphData={graphData} />
          <SignalChart label="LTE" dataKey="LTE" graphData={graphData} />
          <SignalChart label="Radar" dataKey="Radar" graphData={graphData} />
          <SignalChart label="JSSS" dataKey="JSSS" graphData={graphData} />
          <SignalChart label="Total" dataKey="All" graphData={graphData} />
        </div>
      </div>

      {statusMsg && <p style={{ textAlign: "center", padding: "10px 0" }}>{statusMsg}</p>}
    </div>
  );
}
