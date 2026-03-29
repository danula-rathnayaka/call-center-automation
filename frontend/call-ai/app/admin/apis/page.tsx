"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

type ApiItem = {
  id: string;
  name: string;
  apiUrl: string;
  httpMethod: string;
  parameters: string;
};

export default function ApisPage() {
  const router = useRouter();

  const [apis, setApis] = useState<ApiItem[]>([
    {
      id: "1",
      name: "Customer Data API",
      apiUrl: "https://api.group11.com/customer",
      httpMethod: "GET",
      parameters: "",
    },
    {
      id: "2",
      name: "Hotel Data API",
      apiUrl: "https://api.group11.com/hotel",
      httpMethod: "POST",
      parameters: "",
    },
  ]);

  // Add new API
  const handleAdd = () => {
    const newApi: ApiItem = {
      id: Date.now().toString(),
      name: "",
      apiUrl: "",
      httpMethod: "GET",
      parameters: "",
    };
    setApis((prev) => [...prev, newApi]);
  };

  // Delete API
  const handleDelete = (id: string) => {
    setApis((prev) => prev.filter((api) => api.id !== id));
  };

  // Update field
  const handleChange = (id: string, field: keyof ApiItem, value: string) => {
    setApis((prev) =>
      prev.map((api) => (api.id === id ? { ...api, [field]: value } : api)),
    );
  };

  return (
    <div className="flex min-h-screen bg-neutral-50 text-neutral-800">
      <main className="flex-1 p-10 border-x border-neutral-200">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.back()}
              className="p-2 bg-neutral-100 hover:bg-neutral-200 rounded-full"
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
              <h1 className="text-2xl font-bold">API Management</h1>
              <p className="text-sm text-neutral-500">
                Add and manage APIs for the automation system.
              </p>
            </div>
          </div>

          <button
            onClick={handleAdd}
            className="px-4 py-2 bg-black text-white rounded-lg hover:bg-neutral-800"
          >
            + Add API
          </button>
        </div>

        {/* Table */}
        <div className="mt-10 bg-white rounded-xl border border-neutral-200 overflow-hidden">
          {apis.length === 0 ? (
            <div className="p-12 text-center text-neutral-500">
              No APIs added.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-neutral-100 text-neutral-600">
                <tr>
                  <th className="px-4 py-3 text-left">Name</th>
                  <th className="px-4 py-3 text-left">API URL</th>
                  <th className="px-4 py-3 text-left">Method</th>
                  <th className="px-4 py-3 text-left">Parameters</th>
                  <th className="px-4 py-3 text-right">Action</th>
                </tr>
              </thead>

              <tbody>
                {apis.map((api) => (
                  <tr key={api.id} className="border-t border-neutral-200">
                    {/* Name */}
                    <td className="px-4 py-3">
                      <input
                        value={api.name}
                        onChange={(e) =>
                          handleChange(api.id, "name", e.target.value)
                        }
                        className="w-full border border-neutral-200 rounded px-2 py-1"
                        placeholder="API Name"
                      />
                    </td>

                    {/* URL */}
                    <td className="px-4 py-3">
                      <input
                        value={api.apiUrl}
                        onChange={(e) =>
                          handleChange(api.id, "apiUrl", e.target.value)
                        }
                        className="w-full border border-neutral-200 rounded px-2 py-1"
                        placeholder="https://api.example.com"
                      />
                    </td>

                    {/* Method */}
                    <td className="px-4 py-3">
                      <select
                        value={api.httpMethod}
                        onChange={(e) =>
                          handleChange(api.id, "httpMethod", e.target.value)
                        }
                        className="border border-neutral-200 rounded px-2 py-1"
                      >
                        <option>GET</option>
                        <option>POST</option>
                        <option>PUT</option>
                        <option>DELETE</option>
                      </select>
                    </td>

                    {/* Parameters */}
                    <td className="px-4 py-3">
                      <input
                        value={api.parameters}
                        onChange={(e) =>
                          handleChange(api.id, "parameters", e.target.value)
                        }
                        className="w-full border border-neutral-200 rounded px-2 py-1"
                        placeholder='{"key":"value"}'
                      />
                    </td>

                    {/* Action */}
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => handleDelete(api.id)}
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
