"use client";

import { useRouter } from "next/navigation";
import { useState, Fragment } from "react";
import { useNotification } from "@/app/components/notifications/NotificationProvider";

type Parameter = {
  id: string;
  name: string;
  type: string;
  description: string;
};

type ApiItem = {
  id: string;
  toolName: string;
  description: string;
  apiUrl: string;
  httpMethod: string;
  parameters: Parameter[];
  saved: boolean;
  editing: boolean;
};

type IngestingItem = {
  id: string;
  toolName: string;
  apiUrl: string;
};

export default function ApisPage() {
  const router = useRouter();
  const { notify } = useNotification();

  const [apis, setApis] = useState<ApiItem[]>([
    {
      id: "1",
      toolName: "Customer Data API",
      description: "Fetches customer data",
      apiUrl: "https://api.group11.com/customer",
      httpMethod: "GET",
      parameters: [],
      saved: true,
      editing: false,
    },
    {
      id: "2",
      toolName: "Hotel Data API",
      description: "Fetches hotel information",
      apiUrl: "https://api.group11.com/hotel",
      httpMethod: "POST",
      parameters: [],
      saved: true,
      editing: false,
    },
  ]);

  const [ingestingItems, setIngestingItems] = useState<IngestingItem[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const handleAdd = () => {
    setApis((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        toolName: "",
        description: "",
        apiUrl: "",
        httpMethod: "GET",
        parameters: [],
        saved: false,
        editing: false,
      },
    ]);
  };

  const handleDelete = (id: string) => {
    const api = apis.find((a) => a.id === id);
    setApis((prev) => prev.filter((a) => a.id !== id));
    notify({
      title: "API Removed",
      message: `"${api?.toolName || "API"}" has been deleted.`,
      type: "info",
    });
  };

  const handleChange = (id: string, field: keyof ApiItem, value: string) => {
    setApis((prev) =>
      prev.map((a) => (a.id === id ? { ...a, [field]: value } : a)),
    );
  };

  const handleEdit = (id: string) => {
    setApis((prev) =>
      prev.map((a) =>
        a.id === id ? { ...a, editing: true, saved: false } : a,
      ),
    );
  };

  const handleCancelEdit = (id: string, snapshot: ApiItem) => {
    setApis((prev) =>
      prev.map((a) =>
        a.id === id ? { ...snapshot, saved: true, editing: false } : a,
      ),
    );
    setExpandedId(null);
  };

  const handleAddParam = (id: string) => {
    setApis((prev) =>
      prev.map((a) =>
        a.id === id
          ? {
              ...a,
              parameters: [
                ...a.parameters,
                {
                  id: crypto.randomUUID(),
                  name: "",
                  type: "string",
                  description: "",
                },
              ],
            }
          : a,
      ),
    );
  };

  const handleParamChange = (
    apiId: string,
    paramIndex: number,
    field: keyof Parameter,
    value: string,
  ) => {
    setApis((prev) =>
      prev.map((a) =>
        a.id === apiId
          ? {
              ...a,
              parameters: a.parameters.map((p, i) =>
                i === paramIndex ? { ...p, [field]: value } : p,
              ),
            }
          : a,
      ),
    );
  };

  const handleRemoveParam = (apiId: string, paramIndex: number) => {
    setApis((prev) =>
      prev.map((a) =>
        a.id === apiId
          ? {
              ...a,
              parameters: a.parameters.filter((_, i) => i !== paramIndex),
            }
          : a,
      ),
    );
  };

  const handleSave = async (id: string, snapshot?: ApiItem) => {
    const api = apis.find((a) => a.id === id);
    if (!api) return;

    if (!api.toolName.trim()) {
      notify({
        title: "Missing Tool Name",
        message: "Please provide a tool name before saving.",
        type: "error",
      });
      return;
    }

    if (
      !api.apiUrl.startsWith("http://") &&
      !api.apiUrl.startsWith("https://")
    ) {
      notify({
        title: "Invalid URL",
        message: "API URL must start with http:// or https://",
        type: "error",
      });
      return;
    }

    const isEdit = !!snapshot;

    setIngestingItems((prev) => [
      ...prev,
      { id, toolName: api.toolName, apiUrl: api.apiUrl },
    ]);
    setApis((prev) => prev.filter((a) => a.id !== id));
    setExpandedId(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/tools", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_name: api.toolName,
          description: api.description,
          api_url: api.apiUrl,
          http_method: api.httpMethod,
          parameters: api.parameters,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        notify({
          title: isEdit ? "Update Failed" : "Registration Failed",
          message: err.detail || `Could not register "${api.toolName}".`,
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
        title: isEdit ? "API Updated" : "API Registered",
        message: `"${api.toolName}" was successfully ${isEdit ? "updated" : "registered"}.`,
        type: "success",
      });
    } catch (e) {
      notify({
        title: "Network Error",
        message: `Could not reach the server while saving "${api.toolName}".`,
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
                Register and manage API tools for the automation system.
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
              No APIs registered.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-neutral-100 text-neutral-600">
                <tr>
                  <th className="px-4 py-3 text-left">Tool Name</th>
                  <th className="px-4 py-3 text-left">Description</th>
                  <th className="px-4 py-3 text-left">API URL</th>
                  <th className="px-4 py-3 text-left">Method</th>
                  <th className="px-4 py-3 text-left">Params</th>
                  <th className="px-4 py-3 text-right">Action</th>
                </tr>
              </thead>

              <tbody>
                {/* Registering rows with loader */}
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
                          {item.toolName || "Unnamed"}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-blue-500 italic" colSpan={4}>
                      {item.apiUrl} — Registering...
                    </td>
                    <td className="px-4 py-3 text-right text-neutral-400">—</td>
                  </tr>
                ))}

                {/* Editable / saved rows */}
                {apis.map((api) => {
                  const snapshot: ApiItem = {
                    ...api,
                    parameters: api.parameters.map((p) => ({ ...p })),
                  };
                  const isEditable = !api.saved || api.editing;
                  const isExpanded = expandedId === api.id;

                  return (
                    <Fragment key={api.id}>
                      <tr
                        className={`border-t border-neutral-200 ${api.editing ? "bg-amber-50" : ""}`}
                      >
                        {/* Tool Name */}
                        <td className="px-4 py-3">
                          <input
                            value={api.toolName}
                            onChange={(e) =>
                              handleChange(api.id, "toolName", e.target.value)
                            }
                            disabled={!isEditable}
                            className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                            placeholder="e.g. get_customer"
                          />
                        </td>

                        {/* Description */}
                        <td className="px-4 py-3">
                          <input
                            value={api.description}
                            onChange={(e) =>
                              handleChange(
                                api.id,
                                "description",
                                e.target.value,
                              )
                            }
                            disabled={!isEditable}
                            className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                            placeholder="What this tool does"
                          />
                        </td>

                        {/* API URL */}
                        <td className="px-4 py-3">
                          <input
                            value={api.apiUrl}
                            onChange={(e) =>
                              handleChange(api.id, "apiUrl", e.target.value)
                            }
                            disabled={!isEditable}
                            className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                            placeholder="https://api.example.com/endpoint"
                          />
                        </td>

                        {/* Method */}
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

                        {/* Params toggle */}
                        <td className="px-4 py-3">
                          <button
                            onClick={() =>
                              setExpandedId(isExpanded ? null : api.id)
                            }
                            className="flex items-center gap-1 text-neutral-500 hover:text-black transition"
                          >
                            <span className="bg-neutral-100 rounded px-2 py-0.5 text-xs font-mono">
                              {api.parameters.length}
                            </span>
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              height="16px"
                              viewBox="0 -960 960 960"
                              width="16px"
                              fill="currentColor"
                              className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
                            >
                              <path d="M480-360 280-560h400L480-360Z" />
                            </svg>
                          </button>
                        </td>

                        {/* Actions */}
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
                                onClick={() =>
                                  handleCancelEdit(api.id, snapshot)
                                }
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

                      {/* Expanded parameters panel */}
                      {isExpanded && (
                        <tr
                          className={
                            api.editing ? "bg-amber-50" : "bg-neutral-50"
                          }
                        >
                          <td
                            colSpan={6}
                            className="px-6 py-4 border-t border-neutral-100"
                          >
                            <div className="flex items-center justify-between mb-3">
                              <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">
                                Parameters
                              </p>
                              {isEditable && (
                                <button
                                  onClick={() => handleAddParam(api.id)}
                                  className="text-xs text-blue-600 hover:underline"
                                >
                                  + Add Parameter
                                </button>
                              )}
                            </div>

                            {api.parameters.length === 0 ? (
                              <p className="text-xs text-neutral-400 italic">
                                No parameters defined.{" "}
                                {isEditable && (
                                  <button
                                    onClick={() => handleAddParam(api.id)}
                                    className="text-blue-500 hover:underline not-italic"
                                  >
                                    Add one
                                  </button>
                                )}
                              </p>
                            ) : (
                              <div className="space-y-2">
                                <div className="grid grid-cols-[1fr_1fr_2fr_auto] gap-2 text-xs text-neutral-400 font-medium px-1">
                                  <span>Name</span>
                                  <span>Type</span>
                                  <span>Description</span>
                                  <span />
                                </div>

                                {api.parameters.map((param, i) => (
                                  <div
                                    key={i}
                                    className="grid grid-cols-[1fr_1fr_2fr_auto] gap-2 items-center"
                                  >
                                    <input
                                      value={param.name}
                                      onChange={(e) =>
                                        handleParamChange(
                                          api.id,
                                          i,
                                          "name",
                                          e.target.value,
                                        )
                                      }
                                      disabled={!isEditable}
                                      className="border border-neutral-200 rounded px-2 py-1 text-xs disabled:bg-neutral-100 disabled:text-neutral-400"
                                      placeholder="param_name"
                                    />
                                    <select
                                      value={param.type}
                                      onChange={(e) =>
                                        handleParamChange(
                                          api.id,
                                          i,
                                          "type",
                                          e.target.value,
                                        )
                                      }
                                      disabled={!isEditable}
                                      className="border border-neutral-200 rounded px-2 py-1 text-xs disabled:bg-neutral-100 disabled:text-neutral-400"
                                    >
                                      <option value="string">string</option>
                                      <option value="number">number</option>
                                      <option value="boolean">boolean</option>
                                      <option value="integer">integer</option>
                                    </select>
                                    <input
                                      value={param.description}
                                      onChange={(e) =>
                                        handleParamChange(
                                          api.id,
                                          i,
                                          "description",
                                          e.target.value,
                                        )
                                      }
                                      disabled={!isEditable}
                                      className="border border-neutral-200 rounded px-2 py-1 text-xs disabled:bg-neutral-100 disabled:text-neutral-400"
                                      placeholder="What this parameter does"
                                    />
                                    {isEditable && (
                                      <button
                                        onClick={() =>
                                          handleRemoveParam(api.id, i)
                                        }
                                        className="text-red-400 hover:text-red-600 transition"
                                      >
                                        <svg
                                          xmlns="http://www.w3.org/2000/svg"
                                          height="16px"
                                          viewBox="0 -960 960 960"
                                          width="16px"
                                          fill="currentColor"
                                        >
                                          <path d="M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360ZM280-720v520-520Z" />
                                        </svg>
                                      </button>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
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
