import { useState, useEffect, useRef, useCallback } from "react";

// ─── Constants ───────────────────────────────────────────────────────────────
const WS_URL = "ws://localhost:8000/ws";
const API_URL = "http://localhost:8000";
const FRAME_INTERVAL = 80;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "fr", label: "French" },
  { code: "ar", label: "Arabic" },
  { code: "de", label: "German" },
  {code: "am", label: "አማርኛ (Amharic)"}
];

// ─── Drawing helpers ──────────────────────────────────────────────────────────

const POSE_CONNECTIONS = [
  [0, 1], // left shoulder → right shoulder
  [0, 2], // left shoulder → left elbow
  [2, 4], // left elbow   → left wrist
  [1, 3], // right shoulder → right elbow
  [3, 5], // right elbow  → right wrist
];

function drawLine(ctx, p1, p2, color) {
  if (!p1 || !p2) return;
  if (p1.v < 0.3 || p2.v < 0.3) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(p1.sx, p1.sy);
  ctx.lineTo(p2.sx, p2.sy);
  ctx.stroke();
  ctx.restore();
}

function drawJoint(ctx, p, color, r = 5) {
  if (!p || p.v < 0.3) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(p.sx, p.sy, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawSkeleton(ctx, points, color) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = points[a], p2 = points[b];
    if (!p1 || !p2) continue;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  ctx.fillStyle = "#fff";
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// Static fallback body — used for old hand-only format
function drawBody(ctx, w, h) {
  const cx       = w / 2;
  const headY    = h * 0.12;
  const shoulderY = h * 0.30;
  const waistY   = h * 0.62;
  const shoulderW = w * 0.20;

  ctx.strokeStyle = "rgba(160,200,255,0.30)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  ctx.beginPath();
  ctx.arc(cx, headY, w * 0.06, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx, headY + w * 0.06);
  ctx.lineTo(cx, shoulderY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx - shoulderW, shoulderY);
  ctx.lineTo(cx + shoulderW, shoulderY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx, shoulderY);
  ctx.lineTo(cx, waistY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx - shoulderW, shoulderY);
  ctx.lineTo(cx - shoulderW * 1.6, shoulderY + h * 0.18);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx + shoulderW, shoulderY);
  ctx.lineTo(cx + shoulderW * 1.6, shoulderY + h * 0.18);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx - shoulderW * 0.7, waistY);
  ctx.lineTo(cx + shoulderW * 0.7, waistY);
  ctx.stroke();
}

// Scale hand points anchored at a specific wrist screen position
function scaleHandToAnchor(rawPoints, anchorSX, anchorSY, canvasW, canvasH) {
  const xs = rawPoints.map(p => p.x);
  const ys = rawPoints.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const bw = maxX - minX, bh = maxY - minY;
  if (bw === 0 || bh === 0) return rawPoints.map(() => ({ x: anchorSX, y: anchorSY }));

  const scale = Math.min(canvasW * 0.20 / bw, canvasH * 0.28 / bh);

  return rawPoints.map(p => ({
    x: (p.x - minX - bw / 2) * scale + anchorSX,
    y: (p.y - minY - bh / 2) * scale + anchorSY,
  }));
}

// Scale hand points for old hand-only fallback (fixed canvas offsets)
function scaleHandPoints(rawPoints, canvasW, canvasH, offsetX = 0) {
  const xs = rawPoints.map(p => p.x);
  const ys = rawPoints.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const bw = maxX - minX, bh = maxY - minY;
  if (bw === 0 || bh === 0) return rawPoints;

  const scale = Math.min(canvasW * 0.25 / bw, canvasH * 0.30 / bh);
  const targetX = canvasW / 2 + offsetX;
  const targetY = canvasH * 0.58;

  return rawPoints.map(p => ({
    x: (p.x - minX - bw / 2) * scale + targetX,
    y: (p.y - minY - bh / 2) * scale + targetY,
  }));
}

function parseHand(flat42) {
  const pts = [];
  for (let i = 0; i < flat42.length; i += 2)
    pts.push({ x: flat42[i], y: flat42[i + 1] });
  return pts;
}

function interpolate(a, b, t) {
  return a.map((v, i) => v + (b[i] - v) * t);
}
// ── Face drawing ──────────────────────────────────────────────────────────────
// face landmark indices in our saved array:
// 0=nose, 1=leftEyeInner, 2=leftEye, 3=leftEyeOuter
// 4=rightEyeInner, 5=rightEye, 6=rightEyeOuter
// 7=leftEar, 8=rightEar, 9=leftMouth, 10=rightMouth

function drawFace(ctx, faceJoints, canvasW) {
  const [
    nose,
    leftEyeInner, leftEye, leftEyeOuter,
    rightEyeInner, rightEye, rightEyeOuter,
    leftEar, rightEar,
    leftMouth, rightMouth,
  ] = faceJoints;

  // Skip if key points aren't visible
  if (!leftEar || !rightEar || leftEar.z > 0.1 || rightEar.z > 0.1) return;

  ctx.save();

  // ── Face outline — half-oval shape ────────────────────────────────────────
  // Use ears + nose to define the oval proportions
  const faceWidth  = Math.abs(rightEar.sx - leftEar.sx);
  const faceHeight = faceWidth * 1.35; // slightly taller than wide
  const faceCX     = (leftEar.sx + rightEar.sx) / 2;
  // Center the oval so ears touch the sides and nose is roughly in middle
  const faceCY     = nose.sy - faceHeight * 0.1;

  // Outer glow
  ctx.shadowColor = "rgba(160,200,255,0.4)";
  ctx.shadowBlur  = 16;
  ctx.strokeStyle = "rgba(160,200,255,0.55)";
  ctx.lineWidth   = 2;

  // Draw half-oval face: full ellipse but slightly flattened on top
  ctx.beginPath();
  ctx.ellipse(
    faceCX, faceCY,
    faceWidth  * 0.52,  // x-radius — slightly inside the ears
    faceHeight * 0.50,  // y-radius
    0, 0, Math.PI * 2
  );
  ctx.stroke();

  // ── Eyes ─────────────────────────────────────────────────────────────────
  const eyeRadius = faceWidth * 0.07;

  // Left eye (from viewer's perspective = signer's right)
  if (leftEye) {
    ctx.shadowColor = "#63b3ed";
    ctx.shadowBlur  = 8;
    ctx.strokeStyle = "#63b3ed";
    ctx.lineWidth   = 1.5;

    // Eye outline as a small ellipse
    ctx.beginPath();
    ctx.ellipse(leftEye.sx, leftEye.sy, eyeRadius * 1.4, eyeRadius * 0.7, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Iris dot
    ctx.fillStyle = "#63b3ed";
    ctx.beginPath();
    ctx.arc(leftEye.sx, leftEye.sy, eyeRadius * 0.35, 0, Math.PI * 2);
    ctx.fill();
  }

  // Right eye
  if (rightEye) {
    ctx.shadowColor = "#63b3ed";
    ctx.shadowBlur  = 8;
    ctx.strokeStyle = "#63b3ed";
    ctx.lineWidth   = 1.5;

    ctx.beginPath();
    ctx.ellipse(rightEye.sx, rightEye.sy, eyeRadius * 1.4, eyeRadius * 0.7, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#63b3ed";
    ctx.beginPath();
    ctx.arc(rightEye.sx, rightEye.sy, eyeRadius * 0.35, 0, Math.PI * 2);
    ctx.fill();
  }

  // ── Nose bridge ───────────────────────────────────────────────────────────
  if (nose && leftEye && rightEye) {
    const noseBridgeY = (leftEye.sy + rightEye.sy) / 2 + eyeRadius;
    const noseTipX    = nose.sx;
    const noseTipY    = nose.sy;
    const noseW       = faceWidth * 0.10;

    ctx.shadowColor = "rgba(160,200,255,0.3)";
    ctx.shadowBlur  = 4;
    ctx.strokeStyle = "rgba(160,200,255,0.40)";
    ctx.lineWidth   = 1.5;

    // Bridge line
    ctx.beginPath();
    ctx.moveTo(noseTipX, noseBridgeY);
    ctx.lineTo(noseTipX, noseTipY - eyeRadius);
    ctx.stroke();

    // Nose base arc
    ctx.beginPath();
    ctx.arc(noseTipX, noseTipY, noseW, 0.2, Math.PI - 0.2);
    ctx.stroke();
  }

  // ── Mouth ─────────────────────────────────────────────────────────────────
  if (leftMouth && rightMouth) {
    const mouthW  = Math.abs(rightMouth.sx - leftMouth.sx);
    const mouthCX = (leftMouth.sx + rightMouth.sx) / 2;
    const mouthCY = (leftMouth.sy + rightMouth.sy) / 2;

    ctx.shadowColor = "rgba(160,200,255,0.3)";
    ctx.shadowBlur  = 6;
    ctx.strokeStyle = "rgba(160,200,255,0.55)";
    ctx.lineWidth   = 1.5;

    // Mouth as a subtle arc (slight smile curve)
    ctx.beginPath();
    ctx.moveTo(leftMouth.sx, leftMouth.sy);
    ctx.quadraticCurveTo(
      mouthCX, mouthCY + mouthW * 0.25,  // control point — curves down slightly
      rightMouth.sx, rightMouth.sy
    );
    ctx.stroke();
  }

  // ── Eyebrows (estimated above eyes) ──────────────────────────────────────
  if (leftEyeOuter && leftEyeInner) {
    const browY = leftEye ? leftEye.sy - eyeRadius * 1.8 : leftEyeOuter.sy - eyeRadius * 2;
    ctx.shadowColor = "rgba(160,200,255,0.2)";
    ctx.strokeStyle = "rgba(160,200,255,0.35)";
    ctx.lineWidth   = 1.5;
    ctx.lineCap     = "round";

    ctx.beginPath();
    ctx.moveTo(leftEyeOuter.sx, browY + eyeRadius * 0.3);
    ctx.quadraticCurveTo(
      (leftEyeOuter.sx + leftEyeInner.sx) / 2, browY - eyeRadius * 0.2,
      leftEyeInner.sx, browY + eyeRadius * 0.2
    );
    ctx.stroke();
  }

  if (rightEyeOuter && rightEyeInner) {
    const browY = rightEye ? rightEye.sy - eyeRadius * 1.8 : rightEyeOuter.sy - eyeRadius * 2;
    ctx.strokeStyle = "rgba(160,200,255,0.35)";
    ctx.lineWidth   = 1.5;

    ctx.beginPath();
    ctx.moveTo(rightEyeInner.sx, browY + eyeRadius * 0.2);
    ctx.quadraticCurveTo(
      (rightEyeInner.sx + rightEyeOuter.sx) / 2, browY - eyeRadius * 0.2,
      rightEyeOuter.sx, browY + eyeRadius * 0.3
    );
    ctx.stroke();
  }

  ctx.restore();
}

// ── Updated drawPoseAndHands — 135 feature version ───────────────────────────
function drawPoseAndHands(ctx, frame, canvasW, canvasH) {
  // [0–32]  face: 11 landmarks × (x, y, z)
  // [33–50] body: 6 joints   × (x, y, visibility)
  // [51–92] left hand: 42 values (wrist-normalized x,y)
  // [93–134] right hand: 42 values

  // ── Parse face joints ─────────────────────────────────────────────────────
  const faceJoints = [];
  for (let i = 0; i < 33; i += 3) {
    const x = frame[i], y = frame[i + 1], z = frame[i + 2];
    faceJoints.push({
      x, y, z,
      sx: x * canvasW,
      sy: y * canvasH,
    });
  }

  // ── Parse body joints ─────────────────────────────────────────────────────
  const bodyJoints = [];
  for (let i = 33; i < 51; i += 3) {
    bodyJoints.push({
      x:  frame[i],
      y:  frame[i + 1],
      v:  frame[i + 2],
      sx: frame[i]     * canvasW,
      sy: frame[i + 1] * canvasH,
    });
  }
  const [ls, rs, le, re, lw, rw] = bodyJoints;

  // ── Draw face ─────────────────────────────────────────────────────────────
  drawFace(ctx, faceJoints, canvasW);

  // ── Draw torso (estimated from shoulders) ─────────────────────────────────
  if (ls.v > 0.3 && rs.v > 0.3) {
    const midX        = (ls.sx + rs.sx) / 2;
    const midY        = (ls.sy + rs.sy) / 2;
    const shoulderSpan = Math.abs(rs.sx - ls.sx);
    const waistY      = midY + shoulderSpan * 1.0;

    ctx.save();
    ctx.strokeStyle = "rgba(160,200,255,0.35)";
    ctx.lineWidth   = 3;
    ctx.lineCap     = "round";
    ctx.shadowColor = "rgba(99,179,237,0.2)";
    ctx.shadowBlur  = 6;

    // Neck — from midpoint of shoulders upward
    const [nose] = faceJoints;
    if (nose) {
      ctx.beginPath();
      ctx.moveTo(midX, midY);
      ctx.lineTo(nose.sx, nose.sy + (midY - nose.sy) * 0.6);
      ctx.stroke();
    }

    // Torso
    ctx.beginPath();
    ctx.moveTo(midX, midY);
    ctx.lineTo(midX, waistY);
    ctx.stroke();

    // Waist line
    ctx.beginPath();
    ctx.moveTo(midX - shoulderSpan * 0.35, waistY);
    ctx.lineTo(midX + shoulderSpan * 0.35, waistY);
    ctx.stroke();

    ctx.restore();
  }

  // ── Arm bones ─────────────────────────────────────────────────────────────
  drawLine(ctx, ls, rs, "rgba(160,200,255,0.45)");
  drawLine(ctx, ls, le, "#63b3ed");
  drawLine(ctx, le, lw, "#63b3ed");
  drawLine(ctx, rs, re, "#f6ad55");
  drawLine(ctx, re, rw, "#f6ad55");

  [ls, le, lw].forEach(j => drawJoint(ctx, j, "#63b3ed", 6));
  [rs, re, rw].forEach(j => drawJoint(ctx, j, "#f6ad55", 6));

  // ── Hands anchored at wrist positions ────────────────────────────────────
  const leftRaw  = parseHand(frame.slice(51, 93));
  const rightRaw = parseHand(frame.slice(93, 135));

  if (lw.v > 0.3 && leftRaw.some(p => p.x !== 0 || p.y !== 0)) {
    const scaled = scaleHandToAnchor(leftRaw, lw.sx, lw.sy, canvasW, canvasH);
    drawSkeleton(ctx, scaled, "#63b3ed");
  }

  if (rw.v > 0.3 && rightRaw.some(p => p.x !== 0 || p.y !== 0)) {
    const scaled = scaleHandToAnchor(rightRaw, rw.sx, rw.sy, canvasW, canvasH);
    drawSkeleton(ctx, scaled, "#f6ad55");
  }
}

// ── Frame builder — handles both new and old backend formats ─────────────────
// new format: sequences = [{ word, frames: [...], type: "pose+hands" }, ...]
// old format: sequences = [[...], [...]]  (raw arrays)
function buildAllFrames(sequences) {
  const all = [];
  for (const item of sequences) {
    const frames = Array.isArray(item) ? item : item.frames;
    const type   = Array.isArray(item) ? "hands-only" : (item.type || "hands-only");

    for (let i = 0; i < frames.length - 1; i++) {
      all.push({ frame: frames[i], type });
      all.push({ frame: interpolate(frames[i], frames[i + 1], 0.33), type });
      all.push({ frame: interpolate(frames[i], frames[i + 1], 0.66), type });
    }
    if (frames.length > 0)
      all.push({ frame: frames[frames.length - 1], type });
  }
  return all;
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function StatusDot({ connected }) {
  return (
    <span style={{
      display: "inline-block",
      width: 10, height: 10,
      borderRadius: "50%",
      background: connected ? "#22c55e" : "#ef4444",
      boxShadow: connected ? "0 0 8px #22c55e" : "0 0 8px #ef4444",
      marginRight: 8,
      flexShrink: 0,
    }} />
  );
}

function WordChip({ word, index }) {
  return (
    <span style={{
      display: "inline-block",
      background: "rgba(99,179,237,0.15)",
      border: "1px solid rgba(99,179,237,0.4)",
      borderRadius: 8,
      padding: "4px 10px",
      margin: "3px",
      fontSize: 18,
      fontFamily: "'DM Mono', monospace",
      color: "#e2e8f0",
      animation: "chipIn 0.3s ease",
      animationFillMode: "both",
      animationDelay: `${index * 0.04}s`,
    }}>
      {word}
    </span>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontWeight: 600,
      fontSize: 15,
      letterSpacing: "0.04em",
      color: "#94a3b8",
      paddingBottom: 4,
      borderBottom: "1px solid rgba(99,179,237,0.1)",
    }}>
      {children}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const videoRef      = useRef(null);
  const avatarRef     = useRef(null);
  const wsRef         = useRef(null);
  const frameTimerRef = useRef(null);
  const animFrameRef  = useRef(null);
  const lastWordRef   = useRef("");
  const outputLangRef = useRef("en");

  const [connected,    setConnected]    = useState(false);
  const [sentence,     setSentence]     = useState([]);
  const [translated,   setTranslated]   = useState("");
  const [outputLang,   setOutputLang]   = useState("en");

  // Reverse ASL
  const [reverseInput,  setReverseInput]  = useState("");
  const [reverseLang,   setReverseLang]   = useState("en");
  const [reverseStatus, setReverseStatus] = useState("");
  const [isAnimating,   setIsAnimating]   = useState(false);

  // Keep ref in sync with state so WS closure never goes stale
  useEffect(() => { outputLangRef.current = outputLang; }, [outputLang]);

  // ── Translation ──────────────────────────────────────────────────────────
  const translateSentence = useCallback(async (text, lang) => {
    if (!text || lang === "en") { setTranslated(""); return; }
    try {
      const res = await fetch(`${API_URL}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang }),
      });
      const t = await res.json();
      setTranslated(t.translated || "");
    } catch (_) {}
  }, []);

  // ── Frame loop ───────────────────────────────────────────────────────────
  const stopFrameLoop = useCallback(() => {
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
  }, []);

  const startFrameLoop = useCallback((ws) => {
    stopFrameLoop();
    frameTimerRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || ws.readyState !== WebSocket.OPEN) return;

      const canvas = document.createElement("canvas");
      canvas.width  = 320;
      canvas.height = 240;
      canvas.getContext("2d").drawImage(video, 0, 0, 320, 240);

      canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result.split(",")[1];
          if (ws.readyState === WebSocket.OPEN) ws.send(b64);
        };
        reader.readAsDataURL(blob);
      }, "image/jpeg", 0.7);
    }, FRAME_INTERVAL);
  }, [stopFrameLoop]);

  // ── WebSocket ────────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      startFrameLoop(ws);
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.sentence !== undefined) {
        const words = data.sentence.trim().split(" ").filter(Boolean);

        setSentence(prev => {
          const prevStr = prev.join(" ");
          const newStr  = words.join(" ");

          if (prevStr !== newStr && newStr.length > 0) {
            const lang = outputLangRef.current;
            if (lang !== "en") translateSentence(newStr, lang);
            else setTranslated("");
          }

          return words;
        });

        if (data.word && data.word !== lastWordRef.current) {
          lastWordRef.current = data.word;
          playTTS(data.word);
        }
      }
    };

    ws.onclose = () => {
      setConnected(false);
      stopFrameLoop();
      setTimeout(connectWS, 2000);
    };

    ws.onerror = () => ws.close();
  }, [startFrameLoop, stopFrameLoop, translateSentence]);

  // ── Camera + WS init ─────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    });
    connectWS();
    return () => {
      stopFrameLoop();
      wsRef.current?.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── TTS ──────────────────────────────────────────────────────────────────
  const playTTS = (word) => {
    const audio = document.getElementById("tts-audio");
    if (!audio) return;
    audio.src = `${API_URL}/tts?sentence=${encodeURIComponent(word)}`;
    audio.play().catch(() => {});
  };

  // ── Clear ────────────────────────────────────────────────────────────────
  const clearSentence = async () => {
    setSentence([]);
    setTranslated("");
    lastWordRef.current = "";
    try { await fetch(`${API_URL}/clear`, { method: "POST" }); } catch (_) {}
  };

  // ── Reverse ASL ──────────────────────────────────────────────────────────
  const playReverseASL = async () => {
    if (!reverseInput.trim() || isAnimating) return;
    setIsAnimating(true);

    let textToSign = reverseInput.trim();

    if (reverseLang !== "en") {
      setReverseStatus("Translating to English...");
      try {
        const res = await fetch(`${API_URL}/translate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textToSign, lang: "en", source_lang: reverseLang }),
        });
        const t = await res.json();
        textToSign = t.translated || textToSign;
      } catch (_) {}
    }

    setReverseStatus(`Signing: "${textToSign}"`);

    try {
      const res  = await fetch(`${API_URL}/reverse_landmarks?sentence=${encodeURIComponent(textToSign)}`);
      const data = await res.json();

      if (!data.sequences || data.sequences.length === 0) {
        setReverseStatus("⚠ No data found for those words");
        setIsAnimating(false);
        return;
      }

      animateFrames(buildAllFrames(data.sequences), () => {
        setReverseStatus("✓ Done");
        setIsAnimating(false);
      });
    } catch (_) {
      setReverseStatus("Error fetching landmarks");
      setIsAnimating(false);
    }
  };

  // ── Animate frames — dispatches to pose+hands or old hand-only path ──────
  const animateFrames = (frames, onDone) => {
    const canvas = avatarRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    let i = 0;

    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    const draw = () => {
      if (i >= frames.length) { onDone(); return; }

      ctx.clearRect(0, 0, w, h);

      // Subtle background grid
      ctx.strokeStyle = "rgba(99,179,237,0.05)";
      ctx.lineWidth = 1;
      for (let gx = 0; gx < w; gx += 40) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (let gy = 0; gy < h; gy += 40) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }

      const { frame, type } = frames[i];

      if (type === "pose+hands") {
        // New format: 102 values — pose-driven body + anchored hands
        drawPoseAndHands(ctx, frame, w, h);
      } else {
        // Old format: 84 values — static body + scaled hands
        drawBody(ctx, w, h);
        const leftRaw  = parseHand(frame.slice(0, 42));
        const rightRaw = parseHand(frame.slice(42));

        if (leftRaw.some(p => p.x !== 0 || p.y !== 0))
          drawSkeleton(ctx, scaleHandPoints(leftRaw,  w, h, -w * 0.18), "#63b3ed");
        if (rightRaw.some(p => p.x !== 0 || p.y !== 0))
          drawSkeleton(ctx, scaleHandPoints(rightRaw, w, h, +w * 0.18), "#f6ad55");
      }

      i++;
      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

        @keyframes chipIn {
          from { opacity: 0; transform: translateY(6px) scale(0.92); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.5; }
        }

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: #050d1a;
          color: #e2e8f0;
          font-family: 'Syne', sans-serif;
          min-height: 100vh;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
      `}</style>

      <audio id="tts-audio" style={{ display: "none" }} />

      <div style={{
        minHeight: "100vh",
        background: "radial-gradient(ellipse at 20% 20%, rgba(30,60,120,0.6) 0%, transparent 60%), radial-gradient(ellipse at 80% 80%, rgba(10,30,80,0.8) 0%, transparent 60%), #050d1a",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}>

        {/* ── Header ── */}
        <div style={{ textAlign: "center", paddingBottom: 8 }}>
          <h1 style={{
            fontFamily: "'Syne', sans-serif",
            fontWeight: 800,
            fontSize: "clamp(22px, 4vw, 38px)",
            letterSpacing: "-0.02em",
            background: "linear-gradient(135deg, #63b3ed 0%, #a78bfa 50%, #f6ad55 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            ASL Bidirectional Communication
          </h1>
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: 6, marginTop: 8, opacity: 0.7, fontSize: 13,
          }}>
            <StatusDot connected={connected} />
            {connected ? "WebSocket connected — real-time" : "Reconnecting..."}
          </div>
        </div>

        {/* ── Main grid ── */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(480px, 1fr))",
          gap: 20,
        }}>

          {/* ── LEFT: Recognition ── */}
          <div style={panelStyle}>
            <SectionLabel>Live ASL → Text</SectionLabel>

            <div style={{ position: "relative", borderRadius: 12, overflow: "hidden", background: "#000" }}>
              <video
                ref={videoRef}
                autoPlay playsInline muted
                style={{ width: "100%", display: "block", borderRadius: 12 }}
              />
              {!connected && (
                <div style={{
                  position: "absolute", inset: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: "rgba(0,0,0,0.6)",
                  fontSize: 14, color: "#f6ad55",
                  animation: "pulse 1.5s infinite",
                }}>
                  Connecting to server...
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ fontSize: 13, opacity: 0.6, flexShrink: 0 }}>Translate to:</span>
              <select
                value={outputLang}
                onChange={e => setOutputLang(e.target.value)}
                style={selectStyle}
              >
                {LANGUAGES.map(l => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
            </div>

            <div style={boxStyle}>
              <div style={labelStyle}>Recognized sentence</div>
              <div style={{ minHeight: 40, lineHeight: 1.8 }}>
                {sentence.length === 0
                  ? <span style={{ opacity: 0.3, fontFamily: "'DM Mono', monospace", fontSize: 14 }}>Start signing...</span>
                  : sentence.map((w, i) => <WordChip key={i} word={w} index={i} />)
                }
              </div>
            </div>

            {outputLang !== "en" && sentence.length > 0 && (
              <div style={boxStyle}>
                <div style={labelStyle}>
                  Translation ({LANGUAGES.find(l => l.code === outputLang)?.label})
                </div>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 16, color: "#a78bfa", minHeight: 28 }}>
                  {translated
                    ? translated
                    : <span style={{ opacity: 0.4, fontSize: 13 }}>Translating...</span>
                  }
                </div>
              </div>
            )}

            <button onClick={clearSentence} style={btnStyle("#ef4444")}>
              Clear
            </button>
          </div>

          {/* ── RIGHT: Reverse ASL ── */}
          <div style={panelStyle}>
            <SectionLabel>Text → ASL Avatar</SectionLabel>

            <div style={{ display: "flex", gap: 8 }}>
              <select
                value={reverseLang}
                onChange={e => setReverseLang(e.target.value)}
                style={{ ...selectStyle, width: "auto", flexShrink: 0 }}
              >
                {LANGUAGES.map(l => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
              <input
                value={reverseInput}
                onChange={e => setReverseInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && playReverseASL()}
                placeholder="Type a word or sentence..."
                style={inputStyle}
              />
            </div>

            {reverseLang !== "en" && (
              <div style={{ fontSize: 12, opacity: 0.5, paddingLeft: 4 }}>
                Will translate to English first, then sign
              </div>
            )}

            <button
              onClick={playReverseASL}
              disabled={isAnimating || !reverseInput.trim()}
              style={btnStyle(isAnimating ? "#475569" : "#63b3ed", isAnimating)}
            >
              {isAnimating ? "Animating..." : "Animate Sign ▶"}
            </button>

            {reverseStatus && (
              <div style={{ fontSize: 13, color: "#f6ad55", opacity: 0.8, paddingLeft: 4 }}>
                {reverseStatus}
              </div>
            )}

            <div style={{
              position: "relative",
              borderRadius: 12,
              overflow: "hidden",
              background: "#020a15",
              border: "1px solid rgba(99,179,237,0.15)",
            }}>
              <canvas
                ref={avatarRef}
                width={640}
                height={420}
                style={{ width: "100%", display: "block" }}
              />
              {!isAnimating && (
                <div style={{
                  position: "absolute", inset: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  pointerEvents: "none",
                  opacity: 0.15, fontSize: 13,
                  fontFamily: "'DM Mono', monospace",
                }}>
                  Avatar will appear here
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 16, fontSize: 12, opacity: 0.5 }}>
              <span><span style={{ color: "#63b3ed" }}>●</span> Left hand</span>
              <span><span style={{ color: "#f6ad55" }}>●</span> Right hand</span>
              <span><span style={{ color: "rgba(160,200,255,0.5)" }}>—</span> Body</span>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}

// ─── Style constants ──────────────────────────────────────────────────────────
const panelStyle = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(99,179,237,0.1)",
  borderRadius: 16,
  padding: 20,
  display: "flex",
  flexDirection: "column",
  gap: 12,
  backdropFilter: "blur(8px)",
};

const boxStyle = {
  background: "rgba(0,0,0,0.3)",
  border: "1px solid rgba(99,179,237,0.1)",
  borderRadius: 10,
  padding: "12px 14px",
};

const labelStyle = {
  fontSize: 11,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  opacity: 0.45,
  marginBottom: 8,
  fontFamily: "'DM Mono', monospace",
};

const selectStyle = {
  width: "100%",
  background: "rgba(0,0,0,0.4)",
  border: "1px solid rgba(99,179,237,0.2)",
  borderRadius: 8,
  color: "#e2e8f0",
  padding: "8px 12px",
  fontSize: 14,
  fontFamily: "'Syne', sans-serif",
  cursor: "pointer",
  outline: "none",
};

const inputStyle = {
  flex: 1,
  background: "rgba(0,0,0,0.4)",
  border: "1px solid rgba(99,179,237,0.2)",
  borderRadius: 8,
  color: "#e2e8f0",
  padding: "8px 12px",
  fontSize: 14,
  fontFamily: "'Syne', sans-serif",
  outline: "none",
};

const btnStyle = (color, disabled = false) => ({
  background: disabled ? "#1e293b" : `${color}22`,
  border: `1px solid ${disabled ? "#334155" : color}`,
  borderRadius: 8,
  color: disabled ? "#475569" : color,
  padding: "10px 16px",
  fontSize: 14,
  fontFamily: "'Syne', sans-serif",
  fontWeight: 600,
  cursor: disabled ? "not-allowed" : "pointer",
  transition: "all 0.2s",
  letterSpacing: "0.02em",
});
