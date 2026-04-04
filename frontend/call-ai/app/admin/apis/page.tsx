"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useNotification } from "@/app/components/notifications/NotificationProvider";

type ApiItem = {
  id: string;
  name: string;
  apiUrl: string;
  httpMethod: string;
  parameters: string;
  saved: boolean;
  editing: boolean;
};

type IngestingItem = {
  id: string;
  name: string;
  apiUrl: string;
};

export default function ApisPage() {
  const router = useRouter();
  const { notify } = useNotification();

  const [apis, setApis] = useState<ApiItem[]>([
    {
      id: "1",
      name: "Customer Data API",
      apiUrl: "https://api.group11.com/customer",
      httpMethod: "GET",
      parameters: "",
      saved: true,
      editing: false,
    },
    {
      id: "2",
      name: "Hotel Data API",
      apiUrl: "https://api.group11.com/hotel",
      httpMethod: "POST",
      parameters: "",
      saved: true,
      editing: false,
    },
  ]);

  const [ingestingItems, setIngestingItems] = useState<IngestingItem[]>([]);

  const handleAdd = () => {
    const newApi: ApiItem = {
      id: Date.now().toString(),
      name: "",
      apiUrl: "",
      httpMethod: "GET",
      parameters: "",
      saved: false,
      editing: false,
    };
    setApis((prev) => [...prev, newApi]);
  };

  const handleDelete = (id: string) => {
    const api = apis.find((a) => a.id === id);
    setApis((prev) => prev.filter((api) => api.id !== id));
    notify({
      title: "API Removed",
      message: `"${api?.name || "API"}" has been deleted.`,
      type: "info",
    });
  };

  const handleChange = (id: string, field: keyof ApiItem, value: string) => {
    setApis((prev) =>
      prev.map((api) => (api.id === id ? { ...api, [field]: value } : api)),
    );
  };

  const handleEdit = (id: string) => {
    setApis((prev) =>
      prev.map((api) =>
        api.id === id ? { ...api, editing: true, saved: false } : api,
      ),
    );
  };

  const handleCancelEdit = (id: string, snapshot: ApiItem) => {
    setApis((prev) =>
      prev.map((api) =>
        api.id === id ? { ...snapshot, saved: true, editing: false } : api,
      ),
    );
  };

  const handleSave = async (id: string, snapshot?: ApiItem) => {
    const api = apis.find((a) => a.id === id);
    if (!api) return;

    if (
      !api.apiUrl.startsWith("http://") &&
      !api.apiUrl.startsWith("https://")
    ) {
      notify({
        title: "Invalid URL",
        message: "URL must start with http:// or https://",
        type: "error",
      });
      return;
    }

    const isEdit = !!snapshot;

    setIngestingItems((prev) => [
      ...prev,
      { id, name: api.name, apiUrl: api.apiUrl },
    ]);
    setApis((prev) => prev.filter((a) => a.id !== id));

    try {
      const res = await fetch("http://127.0.0.1:8000/api/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: api.apiUrl }),
      });

      if (!res.ok) {
        const err = await res.json();
        notify({
          title: isEdit ? "Update Failed" : "Ingestion Failed",
          message: err.detail || `Could not ingest "${api.name}".`,
          type: "error",
        });
        setApis((prev) => [
          ...prev,
          snapshot
            ? { ...snapshot, saved: true, editing: false }
            : { ...api, saved: false, editing: false },
        ]);
        return;
      }

      setApis((prev) => [...prev, { ...api, saved: true, editing: false }]);
      notify({
        title: isEdit ? "API Updated" : "API Added",
        message: `"${api.name || api.apiUrl}" was successfully ${isEdit ? "updated" : "added"}.`,
        type: "success",
      });
    } catch (e) {
      notify({
        title: "Network Error",
        message: `Could not reach the server while saving "${api.name}".`,
        type: "error",
      });
      setApis((prev) => [
        ...prev,
        snapshot
          ? { ...snapshot, saved: true, editing: false }
          : { ...api, saved: false, editing: false },
      ]);
    } finally {
      setIngestingItems((prev) => prev.filter((i) => i.id !== id));
    }
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
          {apis.length === 0 && ingestingItems.length === 0 ? (
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
                {/* Ingesting rows with loader */}
                {ingestingItems.map((item) => (
                  <tr
                    key={item.id}
                    className="border-t border-neutral-200 bg-blue-50"
                  >
                    <td className="px-4 py-3 font-medium">
                      <div className="flex items-center gap-3">
                        <svg
                          className="animate-spin h-4 w-4 text-blue-500 shrink-0"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                          />
                        </svg>
                        <span className="text-blue-700">
                          {item.name || "Unnamed API"}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-neutral-500">
                      {item.apiUrl}
                    </td>
                    <td className="px-4 py-3 text-neutral-400">—</td>
                    <td className="px-4 py-3 text-blue-500 italic">
                      Processing...
                    </td>
                    <td className="px-4 py-3 text-right text-neutral-400">—</td>
                  </tr>
                ))}

                {/* Editable / saved rows */}
                {apis.map((api) => {
                  const snapshot: ApiItem = { ...api };
                  const isEditable = !api.saved || api.editing;

                  return (
                    <tr
                      key={api.id}
                      className={`border-t border-neutral-200 ${api.editing ? "bg-amber-50" : ""}`}
                    >
                      <td className="px-4 py-3">
                        <input
                          value={api.name}
                          onChange={(e) =>
                            handleChange(api.id, "name", e.target.value)
                          }
                          disabled={!isEditable}
                          className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                          placeholder="API Name"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <input
                          value={api.apiUrl}
                          onChange={(e) =>
                            handleChange(api.id, "apiUrl", e.target.value)
                          }
                          disabled={!isEditable}
                          className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                          placeholder="https://api.example.com"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <select
                          value={api.httpMethod}
                          onChange={(e) =>
                            handleChange(api.id, "httpMethod", e.target.value)
                          }
                          disabled={!isEditable}
                          className="border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                        >
                          <option>GET</option>
                          <option>POST</option>
                          <option>PUT</option>
                          <option>DELETE</option>
                        </select>
                      </td>
                      <td className="px-4 py-3">
                        <input
                          value={api.parameters}
                          onChange={(e) =>
                            handleChange(api.id, "parameters", e.target.value)
                          }
                          disabled={!isEditable}
                          className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                          placeholder='{"key":"value"}'
                        />
                      </td>
                      <td className="px-4 py-3 text-right space-x-3">
                        {api.saved && !api.editing && (
                          <>
                            <button
                              onClick={() => handleEdit(api.id)}
                              className="text-amber-600 hover:underline"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => handleDelete(api.id)}
                              className="text-red-600 hover:underline"
                            >
                              Delete
                            </button>
                          </>
                        )}
                        {api.editing && (
                          <>
                            <button
                              onClick={() => handleSave(api.id, snapshot)}
                              className="text-blue-600 hover:underline"
                            >
                              Save
                            </button>
                            <button
                              onClick={() => handleCancelEdit(api.id, snapshot)}
                              className="text-neutral-500 hover:underline"
                            >
                              Cancel
                            </button>
                          </>
                        )}
                        {!api.saved && !api.editing && (
                          <>
                            <button
                              onClick={() => handleSave(api.id)}
                              className="text-blue-600 hover:underline"
                            >
                              Save
                            </button>
                            <button
                              onClick={() => handleDelete(api.id)}
                              className="text-red-600 hover:underline"
                            >
                              Delete
                            </button>
                          </>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
