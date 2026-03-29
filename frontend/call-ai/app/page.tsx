"use client";

import { useEffect, useRef, useState } from "react";
import CallControls from "./components/CallControls";
import HangOnButton from "./components/HangOnButton";
import { Fragment } from "react";
import { Transition } from "@headlessui/react";
import PhoneNumberDialog from "./components/PhoneNumberDialog";
import { useNotification } from "./components/notifications/NotificationProvider";

declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

export default function Home() {
  const [isListening, setIsListening] = useState(false);
  const [isActiveCall, setActiveCall] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const [isNumberDialogOpen, setNumberDialogOpen] = useState(false);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);

  const recognitionRef = useRef<any>(null);
  const { notify } = useNotification();

  // Audio analysis refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const speak = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);

    utterance.lang = "en-US";
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => {
      setIsAgentSpeaking(true);
    };

    utterance.onend = () => {
      setIsAgentSpeaking(false);
    };

    speechSynthesis.speak(utterance);
  };

  const sendToBackend = async (text: string) => {
    const res = await fetch("/api/agent", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: text, session_id: "123123" }),
    });

    const data = await res.json();

    console.log(data);

    if (data.escalate) {
      if (data.escalate) {
        notify({
          title: "Escalated Call",
          message: "A call requires admin attention",
        });
      }
    }

    speak(data.response);
  };

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
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;

        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        }
      }

      if (finalTranscript) {
        setTranscript(finalTranscript);
        sendToBackend(finalTranscript); // send to backend
      }
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

  const handleActiveCall = () => {
    const phoneNo = sessionStorage.getItem("phone_no");
    if (phoneNo === null) {
      setNumberDialogOpen(true);
      return;
    }
    setActiveCall(true);
  };

  const handleHangOn = (data: boolean) => {
    setActiveCall(data);
  };

  const handlePhoneSubmit = (phone: string) => {
    if (phone.length === 10) {
      sessionStorage.setItem("phone_no", phone);
      handleActiveCall();
    }
  };

  return (
    <>
      <main className="flex min-h-screen flex-col justify-between bg-gradient-to-b from-[#B0EBFF] to-white text-center px-4">
        {/* Title */}
        <div className="pt-20">
          <h1 className="text-3xl font-bold text-black">Call Center Agent</h1>

          <p className="mt-3 text-xl text-gray-800">
            How can I help you today ?
          </p>
        </div>

        <div className="flex items-center justify-center relative">
          {/* Start Call Button */}
          <Transition
            as={Fragment}
            show={!isActiveCall}
            enter="transition ease-out duration-300"
            enterFrom="opacity-0 scale-95"
            enterTo="opacity-100 scale-100"
            leave="transition ease-in duration-200"
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-95"
          >
            <div className="absolute flex items-center justify-center">
              <button
                onClick={handleActiveCall}
                className="relative z-10 w-64 h-64 rounded-full flex items-center justify-center transition-all duration-300 bg-[#008E47] hover:bg-[#006231] shadow-xl"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  height="24px"
                  viewBox="0 -960 960 960"
                  width="24px"
                  fill="#FFFFFF"
                  className="w-24 h-24"
                >
                  <path d="M798-120q-125 0-247-54.5T329-329Q229-429 174.5-551T120-798q0-18 12-30t30-12h162q14 0 25 9.5t13 22.5l26 140q2 16-1 27t-11 19l-97 98q20 37 47.5 71.5T387-386q31 31 65 57.5t72 48.5l94-94q9-9 23.5-13.5T670-390l138 28q14 4 23 14.5t9 23.5v162q0 18-12 30t-30 12ZM241-600l66-66-17-94h-89q5 41 14 81t26 79Zm358 358q39 17 79.5 27t81.5 13v-88l-94-19-67 67ZM241-600Zm358 358Z" />
                </svg>
              </button>
            </div>
          </Transition>

          {/* Active Call UI */}
          <Transition
            as={Fragment}
            show={isActiveCall}
            enter="transition ease-out duration-300"
            enterFrom="opacity-0 scale-95"
            enterTo="opacity-100 scale-100"
            leave="transition ease-in duration-200"
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-95"
          >
            <div className="flex flex-col gap-8 items-center">
              <CallControls
                isListening={isListening}
                isAgentSpeaking={isAgentSpeaking}
                audioLevel={audioLevel}
                onMicClick={handleMicClick}
              />
              <div className="flex flex-col items-center gap-2">
                <HangOnButton onHangOn={handleHangOn} />
                <p className="text-sm text-gray-500">Session Id: 12DWD4W</p>
              </div>
            </div>
          </Transition>
        </div>

        {/* Transcript */}
        {/* {transcript && (
        <div className="mt-10 max-w-xl bg-white/70 backdrop-blur-md p-6 rounded-xl shadow-md">
          <p className="text-gray-800">{transcript}</p>
        </div>
      )} */}

        {/* Footer */}
        <footer className="pb-6 text-sm text-gray-700">
          Powered by Call Center Automation Solution by Group 11
        </footer>
      </main>

      <PhoneNumberDialog
        isOpen={isNumberDialogOpen}
        onClose={() => setNumberDialogOpen(false)}
        onSubmit={handlePhoneSubmit}
      />
    </>
  );
}
