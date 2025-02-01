import { Outlet, RouteObject, createBrowserRouter } from "react-router-dom";
import { Layout } from "./Layout";

export const ROUTES: RouteObject[] = [
  { path: "", lazy: () => import("~/pages/landing") },
  { path: "authorized", lazy: () => import("~/pages/authorized") },
  { path: "onboarding", lazy: () => import("~/pages/onboarding") },
  { path: "home", lazy: () => import("~/pages/home") },
  { path: "ocr", lazy: () => import("~/pages/ocr") },
  { path: "playlist/:playlistId", lazy: () => import("~/pages/playlist") },
];

export const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Layout>
        <Outlet />
      </Layout>
    ),
    children: ROUTES,
  },
]);
