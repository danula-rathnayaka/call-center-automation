"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { useNotification } from "@/app/components/notifications/NotificationProvider";

type UrlItem = {
  id: string;
  name: string;
  url: string;
  saved: boolean;
  editing: boolean;
};

type IngestingItem = {
  id: string;
  name: string;
  url: string;
};

export default function UrlPage() {
  const router = useRouter();
  const { notify } = useNotification();

  const [urls, setUrls] = useState<UrlItem[]>([]);

  const [ingestingItems, setIngestingItems] = useState<IngestingItem[]>([]);

  useEffect(() => {
    const fetchUrls = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/knowledge/urls");

        if (!res.ok) {
          throw new Error("Failed to fetch URLs");
        }

        const data = await res.json();

        const formatted: UrlItem[] = data.urls.map(
          (item: any, index: number) => ({
            id: `${index}-${item.document_hash || item.source}`,
            name: item.source || "Unnamed",
            url: item.source,
            saved: true,
            editing: false,
          }),
        );

        setUrls(formatted);
      } catch (err) {
        notify({
          title: "Error",
          message: "Failed to load URLs from server.",
          type: "error",
        });
      }
    };

    fetchUrls();
  }, []);

  const handleAdd = () => {
    setUrls((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        name: "",
        url: "",
        saved: false,
        editing: false,
      },
    ]);
  };

  const handleDelete = async (id: string, source: string) => {
    const res = await fetch(
      `http://localhost:8000/api/knowledge/urls?source=${source}`,
      {
        method: "DELETE",
      },
    );
    if (!res.ok) {
      const err = await res.json();
      notify({
        title: "Deletation Failed",
        message: err.detail || `Could not delete "${source}".`,
        type: "error",
      });
      return;
    }
    const item = urls.find((u) => u.id === id);
    setUrls((prev) => prev.filter((u) => u.id !== id));
    notify({
      title: "URL Removed",
      message: `"${item?.name || item?.url || "URL"}" has been deleted.`,
      type: "info",
    });
  };

  const handleChange = (id: string, field: keyof UrlItem, value: string) => {
    setUrls((prev) =>
      prev.map((u) => (u.id === id ? { ...u, [field]: value } : u)),
    );
  };

  const handleEdit = (id: string) => {
    setUrls((prev) =>
      prev.map((u) =>
        u.id === id ? { ...u, editing: true, saved: false } : u,
      ),
    );
  };

  const handleCancelEdit = (id: string, snapshot: UrlItem) => {
    setUrls((prev) =>
      prev.map((u) =>
        u.id === id ? { ...snapshot, saved: true, editing: false } : u,
      ),
    );
  };

  const handleSave = async (id: string, snapshot?: UrlItem) => {
    const item = urls.find((u) => u.id === id);
    if (!item) return;

    if (!item.url.startsWith("http://") && !item.url.startsWith("https://")) {
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
      { id, name: item.name, url: item.url },
    ]);
    setUrls((prev) => prev.filter((u) => u.id !== id));

    try {
      const res = await fetch("http://127.0.0.1:8000/api/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: item.url }),
      });

      if (!res.ok) {
        const err = await res.json();
        notify({
          title: isEdit ? "Update Failed" : "Ingestion Failed",
          message: err.detail || `Could not scrape "${item.url}".`,
          type: "error",
        });
        setUrls((prev) => [
          ...prev,
          snapshot
            ? { ...snapshot, saved: true, editing: false }
            : { ...item, saved: false, editing: false },
        ]);
        return;
      }

      setUrls((prev) => [...prev, { ...item, saved: true, editing: false }]);
      notify({
        title: isEdit ? "URL Updated" : "URL Added",
        message: `"${item.name || item.url}" was successfully ${isEdit ? "updated" : "scraped and ingested"}.`,
        type: "success",
      });
    } catch (e) {
      notify({
        title: "Network Error",
        message: `Could not reach the server while saving "${item.url}".`,
        type: "error",
      });
      setUrls((prev) => [
        ...prev,
        snapshot
          ? { ...snapshot, saved: true, editing: false }
          : { ...item, saved: false, editing: false },
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
              <h1 className="text-2xl font-bold">URL Management</h1>
              <p className="text-sm text-neutral-500">
                Add and manage URLs to scrape and ingest data from the web.
              </p>
            </div>
          </div>

          <button
            onClick={handleAdd}
            className="px-4 py-2 bg-black text-white rounded-lg hover:bg-neutral-800"
          >
            + Add URL
          </button>
        </div>

        {/* Table */}
        <div className="mt-10 bg-white rounded-xl border border-neutral-200 overflow-hidden">
          {urls.length === 0 && ingestingItems.length === 0 ? (
            <div className="p-12 text-center text-neutral-500">
              No URLs added.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-neutral-100 text-neutral-600">
                <tr>
                  <th className="px-4 py-3 text-left">Name</th>
                  <th className="px-4 py-3 text-left">URL</th>
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
                          {item.name || "Unnamed"}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-blue-500 italic">
                      {item.url} — Scraping...
                    </td>
                    <td className="px-4 py-3 text-right text-neutral-400">—</td>
                  </tr>
                ))}

                {/* Editable / saved rows */}
                {urls.map((item) => {
                  const snapshot: UrlItem = { ...item };
                  const isEditable = !item.saved || item.editing;

                  return (
                    <tr
                      key={item.id}
                      className={`border-t border-neutral-200 ${item.editing ? "bg-amber-50" : ""}`}
                    >
                      <td className="px-4 py-3">
                        <input
                          value={item.name}
                          onChange={(e) =>
                            handleChange(item.id, "name", e.target.value)
                          }
                          disabled={!isEditable}
                          className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                          placeholder="e.g. Company Blog"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <input
                          value={item.url}
                          onChange={(e) =>
                            handleChange(item.id, "url", e.target.value)
                          }
                          disabled={!isEditable}
                          className="w-full border border-neutral-200 rounded px-2 py-1 disabled:bg-neutral-50 disabled:text-neutral-500"
                          placeholder="https://example.com"
                        />
                      </td>
                      <td className="px-4 py-3 text-right space-x-3">
                        {item.saved && !item.editing && (
                          <>
                            <button
                              onClick={() => handleEdit(item.id)}
                              className="text-amber-600 hover:underline"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => handleDelete(item.id, item.url)}
                              className="text-red-600 hover:underline"
                            >
                              Delete
                            </button>
                          </>
                        )}
                        {item.editing && (
                          <>
                            <button
                              onClick={() => handleSave(item.id, snapshot)}
                              className="text-blue-600 hover:underline"
                            >
                              Save
                            </button>
                            <button
                              onClick={() =>
                                handleCancelEdit(item.id, snapshot)
                              }
                              className="text-neutral-500 hover:underline"
                            >
                              Cancel
                            </button>
                          </>
                        )}
                        {!item.saved && !item.editing && (
                          <>
                            <button
                              onClick={() => handleSave(item.id)}
                              className="text-blue-600 hover:underline"
                            >
                              Save
                            </button>
                            <button
                              onClick={() => handleDelete(item.id, item.url)}
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
