import { Outlet, RouteObject, createBrowserRouter } from "react-router-dom";
import { Layout } from "./Layout";
import { AuthGuard } from "~/components/AuthGuard";

export const ROUTES: RouteObject[] = [
  { path: "", lazy: () => import("~/pages/landing") },
  { path: "onboarding", lazy: () => import("~/pages/onboarding") },
  { path: "home", lazy: () => import("~/pages/home") },
  { path: "ocr", lazy: () => import("~/pages/ocr") },
  { path: "ocr/candidates", lazy: () => import("~/pages/ocr") },
  { path: "playlist/:playlistId", lazy: () => import("~/pages/playlist") },
];

export const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Layout>
        <AuthGuard>
          <Outlet />
        </AuthGuard>
      </Layout>
    ),
    children: ROUTES,
  },
]);
