"use client";

import {
  Dialog,
  DialogPanel,
  DialogTitle,
  Transition,
  TransitionChild,
} from "@headlessui/react";
import { Fragment } from "react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  call: any;
  onEnd: (id: string) => void;
}

export default function CallDetailsDialog({
  isOpen,
  onClose,
  call,
  onEnd,
}: Props) {
  if (!call) return null;

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        {/* Overlay */}
        <TransitionChild as={Fragment}>
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" />
        </TransitionChild>

        <div className="fixed inset-0 flex items-center justify-center p-4">
          <TransitionChild as={Fragment}>
            <DialogPanel className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl">
              <DialogTitle className="text-xl font-semibold text-gray-900">
                Live Call Details
              </DialogTitle>

              {/* Content */}
              <div className="mt-4 space-y-3 text-sm text-gray-700">
                <p>
                  <strong>Phone:</strong> {call.phone_number}
                </p>

                <p>
                  <strong>Customer Intent:</strong> {call.intent}
                </p>

                <p>
                  <strong>Emotion:</strong> {call.emotion}
                </p>

                <p>
                  <strong>Reason:</strong> {call.escalation_reason || "—"}
                </p>

                <div>
                  <strong>Customer Said:</strong>
                  <p className="mt-1 bg-gray-100 p-3 rounded-lg text-gray-800">
                    {call.query}
                  </p>
                </div>
              </div>

              {/* Actions */}
              <div className="mt-6 flex justify-end gap-3">
                <button
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300"
                >
                  Close
                </button>

                <button
                  onClick={() => {
                    onEnd(call.id);
                    onClose();
                  }}
                  className="px-4 py-2 rounded-lg bg-red-600 text-white hover:bg-red-700"
                >
                  End Call
                </button>
              </div>
            </DialogPanel>
          </TransitionChild>
        </div>
      </Dialog>
    </Transition>
  );
}
