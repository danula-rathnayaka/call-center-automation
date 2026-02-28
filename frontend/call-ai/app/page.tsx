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
  const recognitionRef = useRef<any>(null);

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
    };

    recognitionRef.current = recognition;
  }, []);

  const handleMicClick = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setTranscript("");
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-[#B0EBFF] to-white text-center px-4">
      {/* Title */}
      <h1 className="text-3xl font-bold text-black">Call Center Agent</h1>

      <p className="mt-3 text-xl text-gray-800">How can I help you today ?</p>

      {/* Mic Button */}
      <div className="mt-24">
        <button
          onClick={handleMicClick}
          className={`w-64 h-64 rounded-full flex items-center justify-center transition-all duration-300
            ${isListening ? "bg-red-600 scale-105" : "bg-black"}
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
      {transcript && (
        <div className="mt-10 max-w-xl bg-white/70 backdrop-blur-md p-6 rounded-xl shadow-md">
          <p className="text-gray-800">{transcript}</p>
        </div>
      )}

      {/* Footer */}
      <footer className="absolute bottom-6 text-sm text-gray-700">
        Powered by Call Center Automation Solution by Group 12
      </footer>
    </main>
  );
}
