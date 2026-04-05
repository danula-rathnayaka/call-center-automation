export function CallRow({
  phone,
  duration,
  status,
  date,
}: {
  phone: string;
  duration: string;
  status: string;
  date: string;
}) {
  return (
    <tr className="border-t border-neutral-200">
      <td className="px-6 py-4 text-neutral-600">{phone}</td>
      <td className="px-6 py-4">{duration}</td>
      <td className="px-6 py-4">
        <span
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            status === "Completed"
              ? "bg-green-100 text-green-700"
              : "bg-red-100 text-red-700"
          }`}
        >
          {status}
        </span>
      </td>
      <td className="px-6 py-4 text-neutral-500">{date}</td>
    </tr>
  );
}
