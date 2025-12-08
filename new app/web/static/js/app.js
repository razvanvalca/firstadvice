/**
 * Swiss Life Voice Agent Client
 *
 * Handles WebSocket communication, audio streaming, and VAD.
 */

// ============================================
// CONFIGURABLE SETTINGS
// ============================================
const VAD_CONFIG = {
  vadThreshold: 0.08, // Energy threshold for barge-in (0.0-1.0)
  speechThreshold: 0.03, // Energy threshold for speech start detection
  vadDebounceTime: 800, // Debounce time for barge-in (ms)
  silenceCommitDelay: 1200, // Silence duration before committing speech (ms)
};

// ============================================
// VOICE AGENT CLIENT
// ============================================
class VoiceAgentClient {
  constructor() {
    this.ws = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.isConnected = false;
    this.isStreaming = false;

    // Audio capture
    this.scriptProcessor = null;
    this.mediaStreamSource = null;

    // Audio playback
    this.audioQueue = [];
    this.nextPlayTime = 0;
    this.activeSources = [];
    this.gainNode = null;

    // Browser-side VAD
    this.isAgentSpeaking = false;
    this.vadThreshold = VAD_CONFIG.vadThreshold;
    this.vadDebounceTime = VAD_CONFIG.vadDebounceTime;
    this.lastVadTrigger = 0;

    // Silence detection
    this.isSpeaking = false;
    this.silenceStart = 0;
    this.silenceCommitDelay = VAD_CONFIG.silenceCommitDelay;
    this.speechThreshold = VAD_CONFIG.speechThreshold;
  }

  async start() {
    try {
      // Request microphone
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create audio context
      this.audioContext = new AudioContext({ sampleRate: 16000 });

      // Create master gain node for instant muting
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);

      // Connect WebSocket
      await this.connectWebSocket();

      // Start streaming audio
      this.startAudioStream();

      return true;
    } catch (error) {
      console.error("Start error:", error);
      throw error;
    }
  }

  async connectWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log("WebSocket connected");
        this.isConnected = true;
        resolve();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(JSON.parse(event.data));
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        reject(error);
      };

      this.ws.onclose = () => {
        console.log("WebSocket closed");
        this.isConnected = false;
        this.isStreaming = false;
        this.onDisconnect && this.onDisconnect();
      };
    });
  }

  handleMessage(msg) {
    const { type, data, audio } = msg;

    switch (type) {
      case "status":
        this.onStatus && this.onStatus(data);
        break;

      case "partial_transcript":
        this.onPartialTranscript && this.onPartialTranscript(data);
        break;

      case "user_transcript":
        this.onUserTranscript && this.onUserTranscript(data);
        break;

      case "partial_response":
        this.onPartialResponse && this.onPartialResponse(data);
        break;

      case "agent_response":
        this.onAgentResponse && this.onAgentResponse(data);
        break;

      case "audio":
        console.log("Received audio chunk, length:", audio ? audio.length : 0);
        if (audio) {
          this.isAgentSpeaking = true;
          this.queueAudio(audio);
        }
        break;

      case "audio_done":
        console.log("Audio playback complete signal received");
        break;

      case "clear_audio":
        console.log("Clearing audio queue - user interrupted");
        this.clearAudioQueue();
        break;

      case "tasks":
        this.onTasks && this.onTasks(data);
        break;

      case "task_update":
        this.onTaskUpdate && this.onTaskUpdate(data);
        break;

      case "rag_results":
        this.onRagResults && this.onRagResults(data);
        break;

      case "error":
        this.onError && this.onError(data);
        break;
    }
  }

  startAudioStream() {
    this.mediaStreamSource = this.audioContext.createMediaStreamSource(
      this.mediaStream
    );
    this.scriptProcessor = this.audioContext.createScriptProcessor(1024, 1, 1);

    this.scriptProcessor.onaudioprocess = (e) => {
      if (!this.isConnected) return;

      const inputData = e.inputBuffer.getChannelData(0);

      // Calculate audio energy for VAD
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);

      // Barge-in detection
      const now = Date.now();
      if (
        this.isAgentSpeaking &&
        rms > this.vadThreshold &&
        now - this.lastVadTrigger > this.vadDebounceTime
      ) {
        console.log(
          "Browser VAD: User speaking while agent plays - stopping audio! RMS:",
          rms.toFixed(4)
        );
        this.lastVadTrigger = now;
        this.clearAudioQueue();
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(
            JSON.stringify({ type: "user_speaking", interrupted: true })
          );
        }
      }

      // Silence detection for committing transcription
      if (rms > this.speechThreshold) {
        this.isSpeaking = true;
        this.silenceStart = 0;
      } else if (this.isSpeaking) {
        if (this.silenceStart === 0) {
          this.silenceStart = now;
        } else if (now - this.silenceStart > this.silenceCommitDelay) {
          console.log(
            "Browser VAD: Silence detected, committing transcription"
          );
          this.isSpeaking = false;
          this.silenceStart = 0;
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "commit" }));
          }
        }
      }

      // Convert Float32 to Int16 and send
      const int16Data = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      const base64 = this.arrayBufferToBase64(int16Data.buffer);
      this.ws.send(JSON.stringify({ type: "audio", audio: base64 }));
    };

    this.mediaStreamSource.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);
    this.isStreaming = true;
  }

  arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  queueAudio(base64Audio) {
    if (this.audioContext.state === "suspended") {
      this.audioContext.resume();
    }

    const arrayBuffer = this.base64ToArrayBuffer(base64Audio);
    const int16Data = new Int16Array(arrayBuffer);
    const float32Data = new Float32Array(int16Data.length);

    for (let i = 0; i < int16Data.length; i++) {
      float32Data[i] = int16Data[i] / 32768;
    }

    const audioBuffer = this.audioContext.createBuffer(
      1,
      float32Data.length,
      16000
    );
    audioBuffer.getChannelData(0).set(float32Data);

    const currentTime = this.audioContext.currentTime;
    const startTime = Math.max(currentTime, this.nextPlayTime);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.gainNode);
    source.start(startTime);

    this.activeSources.push(source);

    if (this.activeSources.length === 1) {
      this.sendAudioStatus(true);
    }

    source.onended = () => {
      const idx = this.activeSources.indexOf(source);
      if (idx > -1) this.activeSources.splice(idx, 1);

      if (this.activeSources.length === 0) {
        this.isAgentSpeaking = false;
        this.sendAudioStatus(false);
      }
    };

    this.nextPlayTime = startTime + audioBuffer.duration;
    console.log(
      "Queued audio, duration:",
      audioBuffer.duration.toFixed(3),
      "starts at:",
      startTime.toFixed(3)
    );
  }

  sendAudioStatus(playing) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "audio_status", playing: playing }));
      console.log("Audio status:", playing ? "playing" : "stopped");
    }
  }

  clearAudioQueue() {
    console.log("=== CLEARING AUDIO QUEUE ===");
    console.log("Active sources to stop:", this.activeSources.length);

    this.isAgentSpeaking = false;

    if (this.gainNode) {
      this.gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
      console.log("Gain node muted");
    }

    for (const source of this.activeSources) {
      try {
        source.disconnect();
        source.stop(0);
      } catch (e) {
        // Source might already be stopped
      }
    }
    this.activeSources = [];
    this.nextPlayTime = 0;

    if (this.gainNode) {
      this.gainNode.gain.setValueAtTime(1, this.audioContext.currentTime + 0.1);
    }

    console.log("=== AUDIO QUEUE CLEARED ===");
  }

  stop() {
    this.isStreaming = false;
    this.clearAudioQueue();

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.mediaStreamSource) {
      this.mediaStreamSource.disconnect();
      this.mediaStreamSource = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.isConnected = false;
    this.nextPlayTime = 0;
  }
}

// ============================================
// UI CONTROLLER
// ============================================
const client = new VoiceAgentClient();

const orb = document.getElementById("orb");
const status = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const conversation = document.getElementById("conversation");
const messages = document.getElementById("messages");
const partial = document.getElementById("partial");

let currentState = "disconnected";
let hasPartialTranscript = false;
let currentTasks = [];

function setState(state, text) {
  currentState = state;
  status.textContent = text || state;
  status.className = "";
  orb.className = "orb";

  switch (state) {
    case "disconnected":
      orb.innerHTML =
        '🎤<div class="ripple"></div><div class="ripple"></div><div class="ripple"></div>';
      startBtn.style.display = "block";
      stopBtn.style.display = "none";
      break;

    case "connecting":
      orb.innerHTML =
        '⏳<div class="ripple"></div><div class="ripple"></div><div class="ripple"></div>';
      startBtn.disabled = true;
      break;

    case "listening":
      orb.classList.add("listening");
      orb.innerHTML =
        '👂<div class="ripple"></div><div class="ripple"></div><div class="ripple"></div>';
      status.className = "status-listening";
      startBtn.style.display = "none";
      stopBtn.style.display = "block";
      conversation.style.display = "block";
      break;

    case "user-speaking":
      orb.classList.add("user-speaking");
      orb.innerHTML =
        '🗣️<div class="ripple"></div><div class="ripple"></div><div class="ripple"></div>';
      status.className = "status-user-speaking";
      break;

    case "thinking":
      orb.classList.add("thinking");
      orb.innerHTML =
        '🤔<div class="ripple"></div><div class="ripple"></div><div class="ripple"></div>';
      status.className = "status-thinking";
      break;

    case "speaking":
      orb.classList.add("speaking");
      orb.innerHTML =
        '🔊<div class="ripple"></div><div class="ripple"></div><div class="ripple"></div>';
      status.className = "status-speaking";
      break;
  }
}

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="role">${role === "user" ? "You" : "Agent"}</div>
    <div class="content">${content}</div>
  `;
  messages.appendChild(div);
  conversation.scrollTop = conversation.scrollHeight;
}

function renderTasks(tasks) {
  const tasksContainer = document.getElementById("tasksContainer");
  const tasksList = document.getElementById("tasksList");
  const tasksProgressText = document.getElementById("tasksProgressText");
  const tasksProgressFill = document.getElementById("tasksProgressFill");

  if (!tasks || tasks.length === 0) {
    tasksContainer.style.display = "none";
    return;
  }

  currentTasks = tasks;
  tasksContainer.style.display = "block";
  tasksList.innerHTML = "";

  let completedCount = 0;

  tasks.forEach((task) => {
    if (task.completed) completedCount++;

    const taskDiv = document.createElement("div");
    taskDiv.className = `task-item ${task.completed ? "completed" : ""}`;
    taskDiv.id = `task-${task.id}`;
    taskDiv.innerHTML = `
      <div class="task-checkbox"></div>
      <div class="task-description">${task.description}</div>
    `;
    tasksList.appendChild(taskDiv);
  });

  const progressPercent = (completedCount / tasks.length) * 100;
  tasksProgressText.textContent = `${completedCount} of ${tasks.length} completed`;
  tasksProgressFill.style.width = `${progressPercent}%`;
}

function updateTask(taskId, completed) {
  const taskItem = document.getElementById(`task-${taskId}`);
  if (taskItem) {
    if (completed) {
      taskItem.classList.add("completed");
      taskItem.style.transform = "scale(1.02)";
      setTimeout(() => {
        taskItem.style.transform = "";
      }, 200);
    } else {
      taskItem.classList.remove("completed");
    }
  }

  const taskIndex = currentTasks.findIndex((t) => t.id === taskId);
  if (taskIndex !== -1) {
    currentTasks[taskIndex].completed = completed;
  }

  const completedCount = currentTasks.filter((t) => t.completed).length;
  const progressPercent = (completedCount / currentTasks.length) * 100;
  document.getElementById(
    "tasksProgressText"
  ).textContent = `${completedCount} of ${currentTasks.length} completed`;
  document.getElementById(
    "tasksProgressFill"
  ).style.width = `${progressPercent}%`;
}

// Event handlers
client.onStatus = (s) => {
  console.log("Status:", s);
  if (s === "listening") {
    setState("listening", "Listening... just speak naturally");
  } else if (s === "thinking") {
    setState("thinking", "Thinking...");
  } else if (s === "speaking") {
    setState("speaking", "Speaking...");
  } else if (s === "searching") {
    setState("searching", "Searching products...");
  }
};

client.onPartialTranscript = (text) => {
  if (text.trim()) {
    if (!hasPartialTranscript) {
      setState("user-speaking", "Hearing you...");
      hasPartialTranscript = true;
    }
    partial.textContent = `"${text}"`;
    partial.style.display = "block";
    conversation.scrollTop = conversation.scrollHeight;
  }
};

client.onUserTranscript = (text) => {
  hasPartialTranscript = false;
  partial.style.display = "none";
  addMessage("user", text);
};

client.onPartialResponse = (text) => {
  partial.textContent = text;
  partial.style.display = "block";
  conversation.scrollTop = conversation.scrollHeight;
};

client.onAgentResponse = (text) => {
  partial.style.display = "none";
  addMessage("assistant", text);
};

client.onTasks = (tasks) => {
  console.log("Received tasks:", tasks);
  renderTasks(tasks);
};

client.onTaskUpdate = (data) => {
  console.log("Task update:", data);
  updateTask(data.id, data.completed);
};

client.onRagResults = (data) => {
  console.log("RAG results:", data);
  const ragContainer = document.getElementById("ragContainer");
  const ragQuery = document.getElementById("ragQuery");
  const ragResults = document.getElementById("ragResults");

  ragContainer.style.display = "block";
  ragQuery.textContent = `Query: "${data.query}"`;

  ragResults.innerHTML = data.results
    .map(
      (r) => `
    <div class="rag-result">
      <div class="rag-result-header">
        <span class="rag-result-name">${r.product}</span>
        <span class="rag-result-score">Score: ${r.score}</span>
      </div>
      <div class="rag-result-snippet">${r.snippet}</div>
    </div>
  `
    )
    .join("");
};

client.onError = (error) => {
  console.error("Error:", error);
  status.textContent = `Error: ${error}`;
  setTimeout(() => {
    if (currentState !== "disconnected") {
      setState("listening", "Listening... just speak naturally");
    }
  }, 3000);
};

client.onDisconnect = () => {
  setState("disconnected", 'Disconnected - Click "Start" to begin');
  startBtn.disabled = false;
};

// Start button
startBtn.addEventListener("click", async () => {
  try {
    setState("connecting", "Connecting...");
    await client.start();

    const config = {
      type: "config",
      tasks: [
        {
          id: 1,
          description: "Begrüssung und Vorstellung als Swiss Life Berater",
        },
        { id: 2, description: "Namen des Kunden erfragen" },
        {
          id: 3,
          description: "Versicherungssituation und finanzielle Ziele verstehen",
        },
        {
          id: 4,
          description: "Passendes Swiss Life Produkt empfehlen und erklären",
        },
        { id: 5, description: "Interesse an Angebot bestätigen" },
        { id: 6, description: "Notwendige Daten für Angebot erfassen" },
        { id: 7, description: "Nächste Schritte erklären und verabschieden" },
      ],
    };

    setTimeout(() => {
      if (client.ws && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(JSON.stringify(config));
      }
    }, 500);
  } catch (error) {
    setState("disconnected", `Error: ${error.message}`);
    startBtn.disabled = false;
  }
});

// Stop button
stopBtn.addEventListener("click", () => {
  client.stop();
  setState("disconnected", 'Click "Start" to begin');
});
