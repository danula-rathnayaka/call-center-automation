import React from "react";

interface Props {
  isListening: boolean;
  isAgentSpeaking: boolean;
  audioLevel: number;
  onMicClick: () => void;
}

export default function CallControls({
  isListening,
  isAgentSpeaking,
  audioLevel,
  onMicClick,
}: Props) {
  const botWave = isAgentSpeaking ? 30 + Math.sin(Date.now() / 150) * 20 : 0;

  return (
    <div className="flex gap-25">
      {/* USER MIC */}
      <div className="relative flex items-center justify-center">
        {isListening && (
          <>
            <div
              className="absolute w-64 h-64 bg-black rounded-full opacity-10 transition-transform duration-75"
              style={{
                transform: `scale(${1 + Math.min(audioLevel / 50, 0.15)})`,
              }}
            />
            <div
              className="absolute w-64 h-64 border-2 border-black rounded-full opacity-10 transition-transform duration-75"
              style={{
                transform: `scale(${1 + Math.min(audioLevel / 35, 0.3)})`,
              }}
            />
            <div
              className="absolute w-64 h-64 border border-black rounded-full opacity-5 transition-transform duration-75"
              style={{
                transform: `scale(${1 + Math.min(audioLevel / 20, 0.5)})`,
              }}
            />
          </>
        )}

        <button
          onClick={onMicClick}
          className={`relative z-10 w-64 h-64 rounded-full flex items-center justify-center transition-all duration-300
          ${
            isListening
              ? "bg-black scale-[1.02] shadow-[0_0_20px_rgba(0,0,0,0.3)]"
              : "bg-black"
          }
          shadow-xl`}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            fill="#FFFFFF"
            className="w-24 h-24"
          >
            <path d="M395-435q-35-35-35-85v-240q0-50 35-85t85-35q50 0 85 35t35 85v240q0 50-35 85t-85 35q-50 0-85-35Zm85-205Zm-40 520v-123q-104-14-172-93t-68-184h80q0 83 58.5 141.5T480-320q83 0 141.5-58.5T680-520h80q0 105-68 184t-172 93v123h-80Z" />
          </svg>
        </button>
      </div>

      {/* BOT AGENT */}
      <div className="relative flex items-center justify-center">
        {isAgentSpeaking && (
          <>
            <div
              className="absolute w-64 h-64 bg-[#000C22] rounded-full opacity-20 transition-transform duration-100"
              style={{
                transform: `scale(${1 + botWave / 200})`,
              }}
            />
            <div
              className="absolute w-64 h-64 border-2 border-[#000C22] rounded-full opacity-20 transition-transform duration-100"
              style={{
                transform: `scale(${1 + botWave / 120})`,
              }}
            />
            <div
              className="absolute w-64 h-64 border border-[#000C22] rounded-full opacity-10 transition-transform duration-100"
              style={{
                transform: `scale(${1 + botWave / 80})`,
              }}
            />
          </>
        )}

        <button
          className={`relative z-10 w-64 h-64 rounded-full flex items-center justify-center transition-all duration-300
          ${
            isAgentSpeaking
              ? "bg-[#000C22] scale-[1.02] shadow-[0_0_20px_rgba(0,0,0,0.3)]"
              : "bg-[#000C22]"
          }
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
            <path d="M160-120v-200q0-33 23.5-56.5T240-400h480q33 0 56.5 23.5T800-320v200H160Zm200-320q-83 0-141.5-58.5T160-640q0-83 58.5-141.5T360-840h240q83 0 141.5 58.5T800-640q0 83-58.5 141.5T600-440H360ZM240-200h480v-120H240v120Zm120-320h240q50 0 85-35t35-85q0-50-35-85t-85-35H360q-50 0-85 35t-35 85q0 50 35 85t85 35Zm28.5-91.5Q400-623 400-640t-11.5-28.5Q377-680 360-680t-28.5 11.5Q320-657 320-640t11.5 28.5Q343-600 360-600t28.5-11.5Zm240 0Q640-623 640-640t-11.5-28.5Q617-680 600-680t-28.5 11.5Q560-657 560-640t11.5 28.5Q583-600 600-600t28.5-11.5ZM480-200Zm0-440Z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
