import Link from "next/link";

export default function NotFound() {
  return (
    <main className="grid min-h-[100dvh] place-items-center bg-paper px-4 text-ink">
      <div className="w-full max-w-sm rounded-md border border-linen bg-surface p-6 text-center shadow-hairline">
        <p className="font-mono text-sm font-medium text-soot">404</p>
        <h1 className="mt-2 text-xl font-semibold">Page not found</h1>
        <Link
          className="mt-5 inline-flex h-10 items-center rounded-md bg-cobalt px-4 text-sm font-medium text-white transition hover:bg-cobalt-deep"
          href="/"
        >
          Return home
        </Link>
      </div>
    </main>
  );
}
