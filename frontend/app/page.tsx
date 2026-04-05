"use client";

import { useEffect, useRef, useState } from "react";
import CallControls from "./components/CallControls";
import HangOnButton from "./components/HangOnButton";
import { Fragment } from "react";
import { Transition } from "@headlessui/react";
import PhoneNumberDialog from "./components/PhoneNumberDialog";

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
  const [agentTranscript, setAgentTranscript] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);
  const [isNumberDialogOpen, setNumberDialogOpen] = useState(false);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");

  // Audio analysis refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // MediaRecorder refs for voice capture
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const speak = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.onstart = () => setIsAgentSpeaking(true);
    utterance.onend = () => setIsAgentSpeaking(false);
    speechSynthesis.speak(utterance);
  };

  const encodeWAV = (samples: Float32Array, sampleRate: number): Blob => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++)
        view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, "data");
    view.setUint32(40, samples.length * 2, true);

    // Convert float32 → int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  };

  // now accepts a Blob and sends as multipart/form-data
  const sendToBackend = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    formData.append("session_id", "123123");

    const res = await fetch(
      `http://127.0.0.1:8000/api/chat/voice?session_id=${sessionId}`,
      {
        method: "POST",
        body: formData,
      },
    );

    const data = await res.json();
    console.log(data);
    setAgentTranscript(data.response);
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
    setAudioLevel(0);
  };

  // mic click starts/stops MediaRecorder
  const handleMicClick = async () => {
    if (isListening) {
      // Stop recording
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop(); // triggers onstop →encodes real WAV
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      stopAudioAnalysis();
      setIsListening(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        streamRef.current = stream;

        // Set up AudioContext for both analysis and raw PCM capture
        const audioContext = new (
          window.AudioContext || (window as any).webkitAudioContext
        )();
        audioContextRef.current = audioContext;

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        // Analyser for audio level visualisation
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(
          bufferLength,
        ) as Uint8Array<ArrayBuffer>;
        dataArrayRef.current = dataArray;

        const updateAudioLevel = () => {
          if (!analyserRef.current || !dataArrayRef.current) return;
          analyserRef.current.getByteFrequencyData(dataArrayRef.current);
          let sum = 0;
          for (let i = 0; i < dataArrayRef.current.length; i++)
            sum += dataArrayRef.current[i];
          setAudioLevel(sum / dataArrayRef.current.length);
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        };
        updateAudioLevel();

        // ScriptProcessor to collect raw PCM samples
        const sampleRate = audioContext.sampleRate;
        const bufferSize = 4096;
        const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
        const pcmChunks: Float32Array[] = [];

        processor.onaudioprocess = (e) => {
          const channelData = e.inputBuffer.getChannelData(0);
          pcmChunks.push(new Float32Array(channelData));
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        // Store a "fake" stop handler on mediaRecorderRef
        (mediaRecorderRef.current as any) = {
          stop: () => {
            processor.disconnect();
            source.disconnect(processor);

            // Merge all PCM chunks
            const totalLength = pcmChunks.reduce((acc, c) => acc + c.length, 0);
            const merged = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of pcmChunks) {
              merged.set(chunk, offset);
              offset += chunk.length;
            }

            const wavBlob = encodeWAV(merged, sampleRate);
            sendToBackend(wavBlob);
          },
        };

        setIsListening(true);
      } catch (err) {
        console.error("Microphone access error:", err);
      }
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

  const handleHangOn = async (data: boolean) => {
    const res = await fetch(
      `http://127.0.0.1:8000/api/session/end/${sessionId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      },
    );
    const resData = await res.json();

    if (resData.session_id == sessionId) {
      setSessionId("");
      setActiveCall(data);
      sessionStorage.removeItem("phone_no");
    }
  };

  const handlePhoneSubmit = async (phone: string) => {
    if (phone.length === 10) {
      const num: number = Math.floor(100000 + Math.random() * 900000);
      const res = await fetch(
        `http://127.0.0.1:8000/api/session/start?session_id=${num}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ phone_number: phone }),
        },
      );
      const data = await res.json();

      if (data.session_id == num) {
        setSessionId(data.session_id);
        sessionStorage.setItem("phone_no", phone);
        handleActiveCall();
      }
    }
  };

  return (
    <>
      <main className="flex min-h-screen flex-col justify-between bg-gradient-to-b from-[#B0EBFF] to-white text-center px-4">
        <div className="pt-20">
          <h1 className="text-3xl font-bold text-black">Call Center Agent</h1>
          <p className="mt-3 text-xl text-gray-800">
            How can I help you today ?
          </p>
        </div>

        <div className="flex items-center justify-center relative">
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
                {agentTranscript && (
                  <div className="mt-10 max-w-xl bg-[#000C22]/10 backdrop-blur-md p-6 rounded-xl shadow-md mb-4">
                    <p className="text-gray-800">{agentTranscript}</p>
                  </div>
                )}
                <HangOnButton onHangOn={handleHangOn} />
                <p className="text-sm text-gray-500">Session Id: {sessionId}</p>
              </div>
            </div>
          </Transition>
        </div>

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
