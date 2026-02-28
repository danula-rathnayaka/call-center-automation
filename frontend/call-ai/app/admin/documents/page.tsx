"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

type DocumentItem = {
  id: string;
  name: string;
  size: number;
  uploadedAt: string;
};

export default function DocumentsPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // existing documents
  const [documents, setDocuments] = useState<DocumentItem[]>([
    {
      id: "1",
      name: "Customer_Policy_Guide.pdf",
      size: 245760,
      uploadedAt: "2026-02-20 10:15 AM",
    },
    {
      id: "2",
      name: "Call_Script_Template.docx",
      size: 102400,
      uploadedAt: "2026-02-18 02:40 PM",
    },
  ]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFiles = (files: FileList | null) => {
    if (!files) return;

    const newDocs: DocumentItem[] = Array.from(files).map((file) => ({
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      uploadedAt: new Date().toLocaleString(),
    }));

    setDocuments((prev) => [...newDocs, ...prev]);
  };

  const handleDelete = (id: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
  };

  return (
    <div className="flex min-h-screen bg-neutral-50 text-neutral-800">
      <main className="flex-1 p-10 border-x border-neutral-200">
        {/* Page Title */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.back()}
            className="p-2 bg-neutral-100 hover:bg-neutral-200 transition rounded-full cursor-pointer"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#000000"
            >
              <path d="M560-240 320-480l240-240 56 56-184 184 184 184-56 56Z" />
            </svg>
          </button>
          <div>
            <h1 className="text-2xl font-bold">Document Management</h1>
            <p className="text-sm text-neutral-500 mt-1">
              Upload and manage knowledge base documents for the automation
              system.
            </p>
          </div>
        </div>

        {/* Upload Box */}
        <div
          onClick={handleUploadClick}
          className="mt-8 border-2 border-dashed border-neutral-300 rounded-xl p-12 text-center bg-white cursor-pointer hover:border-black transition"
        >
          <p className="font-medium">
            Click to upload or drag & drop documents
          </p>
          <p className="text-xs text-neutral-400 mt-2">
            PDF, DOCX, TXT, CSV supported
          </p>

          <input
            type="file"
            multiple
            ref={fileInputRef}
            onChange={(e) => handleFiles(e.target.files)}
            className="hidden"
          />
        </div>

        {/* Existing Documents - Directly Below Upload */}
        <div className="mt-10 bg-white rounded-xl border border-neutral-200 overflow-hidden">
          {documents.length === 0 ? (
            <div className="p-12 text-center text-neutral-500">
              No documents available.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-neutral-100 text-neutral-600">
                <tr>
                  <th className="text-left px-6 py-4">File Name</th>
                  <th className="text-left px-6 py-4">Size</th>
                  <th className="text-left px-6 py-4">Uploaded</th>
                  <th className="text-left px-6 py-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr
                    key={doc.id}
                    className="border-t border-neutral-200 hover:bg-neutral-50 transition"
                  >
                    <td className="px-6 py-4 font-medium">{doc.name}</td>
                    <td className="px-6 py-4 text-neutral-600">
                      {(doc.size / 1024).toFixed(1)} KB
                    </td>
                    <td className="px-6 py-4 text-neutral-600">
                      {doc.uploadedAt}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={() => handleDelete(doc.id)}
                        className="text-red-600 hover:underline"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
