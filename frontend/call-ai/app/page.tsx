"use client";

import { useEffect, useRef, useState } from "react";

declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

export default function Home() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);

  const recognitionRef = useRef<any>(null);

  // Audio analysis refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopAudioAnalysis = () => {
    if (animationFrameRef.current)
      cancelAnimationFrame(animationFrameRef.current);
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch (e) {}
    }
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch (e) {}
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      try {
        audioContextRef.current.close();
      } catch (e) {}
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    setAudioLevel(0);
  };

  const startAudioAnalysis = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const audioContext = new (
        window.AudioContext || (window as any).webkitAudioContext
      )();
      audioContextRef.current = audioContext;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      sourceRef.current = source;
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      dataArrayRef.current = dataArray;

      const updateAudioLevel = () => {
        if (!analyserRef.current || !dataArrayRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArrayRef.current as any);
        let sum = 0;
        for (let i = 0; i < dataArrayRef.current.length; i++) {
          sum += dataArrayRef.current[i];
        }
        const average = sum / dataArrayRef.current.length;
        setAudioLevel(average);
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      };

      updateAudioLevel();
    } catch (err) {
      console.error("Error accessing microphone for volume analysis", err);
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech Recognition not supported in this browser");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: any) => {
      let currentTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        currentTranscript += event.results[i][0].transcript;
      }

      setTranscript(currentTranscript);
    };

    recognition.onend = () => {
      setIsListening(false);
      stopAudioAnalysis();
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {}
      }
      stopAudioAnalysis();
    };
  }, []);

  const handleMicClick = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      stopAudioAnalysis();
    } else {
      setTranscript("");
      recognitionRef.current.start();
      setIsListening(true);
      startAudioAnalysis();
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-[#B0EBFF] to-white text-center px-4">
      {/* Title */}
      <h1 className="text-3xl font-bold text-black">Call Center Agent</h1>

      <p className="mt-3 text-xl text-gray-800">How can I help you today ?</p>

      {/* Mic Button */}
      <div className="mt-24 relative flex items-center justify-center">
        {/* React to Audio Level effect waves */}
        {isListening && (
          <>
            <div
              className="absolute w-64 h-64 bg-black rounded-full opacity-10 transition-transform duration-75 ease-out"
              style={{
                transform: `scale(${1 + Math.min(audioLevel / 50, 0.15)})`,
              }}
            ></div>
            <div
              className="absolute w-64 h-64 border-2 border-black rounded-full opacity-10 transition-transform duration-75 ease-out"
              style={{
                transform: `scale(${1 + Math.min(audioLevel / 35, 0.3)})`,
              }}
            ></div>
            <div
              className="absolute w-64 h-64 border border-black rounded-full opacity-5 transition-transform duration-75 ease-out"
              style={{
                transform: `scale(${1 + Math.min(audioLevel / 20, 0.5)})`,
              }}
            ></div>
          </>
        )}
        <button
          onClick={handleMicClick}
          className={`relative z-10 w-64 h-64 rounded-full flex items-center justify-center transition-all duration-300
            ${isListening ? "bg-gray-900 scale-[1.02] shadow-[0_0_20px_rgba(0,0,0,0.3)]" : "bg-black"}
            shadow-xl`}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="24px"
            viewBox="0 -960 960 960"
            width="24px"
            fill="#FFFFFF"
            className="w-24 h-24"
          >
            <path d="M395-435q-35-35-35-85v-240q0-50 35-85t85-35q50 0 85 35t35 85v240q0 50-35 85t-85 35q-50 0-85-35Zm85-205Zm-40 520v-123q-104-14-172-93t-68-184h80q0 83 58.5 141.5T480-320q83 0 141.5-58.5T680-520h80q0 105-68 184t-172 93v123h-80Zm68.5-371.5Q520-503 520-520v-240q0-17-11.5-28.5T480-800q-17 0-28.5 11.5T440-760v240q0 17 11.5 28.5T480-480q17 0 28.5-11.5Z" />
          </svg>
        </button>
      </div>

      {/* Transcript */}
      {/* {transcript && (
        <div className="mt-10 max-w-xl bg-white/70 backdrop-blur-md p-6 rounded-xl shadow-md">
          <p className="text-gray-800">{transcript}</p>
        </div>
      )} */}

      {/* Footer */}
      <footer className="absolute bottom-6 text-sm text-gray-700">
        Powered by Call Center Automation Solution by Group 12
      </footer>
    </main>
  );
}
