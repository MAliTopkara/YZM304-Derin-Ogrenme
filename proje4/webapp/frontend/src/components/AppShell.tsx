/** Tüm sayfaları saran üst-bar + dikey container. */
import { Brain } from "lucide-react";
import { type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

import { cn } from "../lib/utils";

const STEPS: { path: string; label: string; index: number }[] = [
  { path: "/", label: "Görsel Yükle", index: 1 },
  { path: "/model", label: "Model Seç", index: 2 },
  { path: "/results", label: "Sonuçlar", index: 3 },
];

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  const currentStep =
    STEPS.find((s) => s.path === location.pathname)?.index ?? 0;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-black/5 bg-white/60 backdrop-blur-sm">
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="size-9 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
              <Brain className="size-5 text-white" />
            </div>
            <div className="leading-tight">
              <div className="font-semibold text-slate-800">
                Gameplay Classifier
              </div>
              <div className="text-xs text-slate-500">
                3 model — ResNet50 · EfficientNetB0 · ViT-Base/16
              </div>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-2">
            {STEPS.map((step) => {
              const active = currentStep === step.index;
              const past = currentStep > step.index;
              return (
                <div
                  key={step.path}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-colors",
                    active && "bg-brand-100 text-brand-700 font-medium",
                    past && "text-slate-500",
                    !active && !past && "text-slate-400",
                  )}
                >
                  <span
                    className={cn(
                      "size-5 rounded-full flex items-center justify-center text-xs font-semibold",
                      active && "bg-brand-500 text-white",
                      past && "bg-emerald-500 text-white",
                      !active && !past && "bg-slate-200 text-slate-500",
                    )}
                  >
                    {past ? "✓" : step.index}
                  </span>
                  {step.label}
                </div>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-6 py-8">{children}</div>
      </main>

      <footer className="border-t border-black/5 py-4 text-center text-xs text-slate-500">
        YZM304 Proje 4 · Lokal demo · CNN ve ViT karşılaştırması
      </footer>
    </div>
  );
}
