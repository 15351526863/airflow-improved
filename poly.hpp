#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <stack>
#include <limits>

struct Vector
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vector() = default;
    Vector(double _x, double _y, double _z = 0.0) : x(_x), y(_y), z(_z) {}

    double DistToSqr(const Vector& other) const
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    bool operator==(const Vector& other) const;
};

namespace poly_intersect
{
    constexpr double equity_tolerance = 1e-9;

    inline bool is_equal(double d1, double d2)
    {
        return std::fabs(d1 - d2) <= equity_tolerance;
    }
}

inline bool Vector::operator==(const Vector& other) const
{
    return poly_intersect::is_equal(x, other.x) &&
        poly_intersect::is_equal(y, other.y) &&
        poly_intersect::is_equal(z, other.z);
}

namespace poly_intersect
{
    using convex_polygon = std::vector<Vector>;

    inline bool inside_poly(const Vector& test, const convex_polygon& poly)
    {
        if (poly.size() < 3) return false;
        bool res = false;
        for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
        {
            double dy = poly[j].y - poly[i].y;
            if (is_equal(dy, 0.0)) continue;
            if (((poly[i].y > test.y) != (poly[j].y > test.y)) &&
                (test.x < (poly[j].x - poly[i].x) * (test.y - poly[i].y) / dy + poly[i].x))
                res = !res;
        }
        return res;
    }

    inline bool point_on_segment(const Vector& p, const Vector& a, const Vector& b)
    {
        return is_equal((p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x), 0.0) &&
            p.x >= std::min(a.x, b.x) - equity_tolerance &&
            p.x <= std::max(a.x, b.x) + equity_tolerance &&
            p.y >= std::min(a.y, b.y) - equity_tolerance &&
            p.y <= std::max(a.y, b.y) + equity_tolerance;
    }

    inline std::optional<Vector> get_intersection(
        const Vector& l1p1, const Vector& l1p2,
        const Vector& l2p1, const Vector& l2p2)
    {
        const double a1 = l1p2.y - l1p1.y;
        const double b1 = l1p1.x - l1p2.x;
        const double c1 = a1 * l1p1.x + b1 * l1p1.y;

        const double a2 = l2p2.y - l2p1.y;
        const double b2 = l2p1.x - l2p2.x;
        const double c2 = a2 * l2p1.x + b2 * l2p1.y;

        const double det = a1 * b2 - a2 * b1;

        if (is_equal(det, 0.0))
        {
            if (!is_equal((a1 * l2p1.x + b1 * l2p1.y), c1)) return std::nullopt;
            if (point_on_segment(l2p1, l1p1, l1p2)) return l2p1;
            if (point_on_segment(l2p2, l1p1, l1p2)) return l2p2;
            if (point_on_segment(l1p1, l2p1, l2p2)) return l1p1;
            if (point_on_segment(l1p2, l2p1, l2p2)) return l1p2;
            return std::nullopt;
        }

        const double x = (b2 * c1 - b1 * c2) / det;
        const double y = (a1 * c2 - a2 * c1) / det;

        const bool online1 = x >= std::min(l1p1.x, l1p2.x) - equity_tolerance &&
            x <= std::max(l1p1.x, l1p2.x) + equity_tolerance &&
            y >= std::min(l1p1.y, l1p2.y) - equity_tolerance &&
            y <= std::max(l1p1.y, l1p2.y) + equity_tolerance;

        const bool online2 = x >= std::min(l2p1.x, l2p2.x) - equity_tolerance &&
            x <= std::max(l2p1.x, l2p2.x) + equity_tolerance &&
            y >= std::min(l2p1.y, l2p2.y) - equity_tolerance &&
            y <= std::max(l2p1.y, l2p2.y) + equity_tolerance;

        if (!(online1 && online2)) return std::nullopt;

        double t = 0.0;
        if (!is_equal(l1p2.x, l1p1.x))
            t = (x - l1p1.x) / (l1p2.x - l1p1.x);
        else if (!is_equal(l1p2.y, l1p1.y))
            t = (y - l1p1.y) / (l1p2.y - l1p1.y);

        t = std::clamp(t, 0.0, 1.0);
        const double z = l1p1.z + (l1p2.z - l1p1.z) * t;

        return Vector(x, y, z);
    }

    inline convex_polygon get_intersections(
        const Vector& l1p1, const Vector& l1p2, const convex_polygon& poly)
    {
        if (poly.size() < 2) return {};
        convex_polygon res;
        res.reserve(poly.size());
        for (size_t i = 0; i < poly.size(); ++i)
        {
            size_t nxt = (i + 1) % poly.size();
            if (auto p = get_intersection(l1p1, l1p2, poly[i], poly[nxt]); p) res.push_back(*p);
        }
        return res;
    }

    inline void add_points(convex_polygon& points, const Vector& np)
    {
        for (const auto& p : points)
            if (is_equal(p.x, np.x) && is_equal(p.y, np.y) && is_equal(p.z, np.z)) return;
        points.push_back(np);
    }

    inline void add_points(convex_polygon& points, const convex_polygon& newpoints)
    {
        points.reserve(points.size() + newpoints.size());
        for (const auto& p : newpoints) add_points(points, p);
    }

    inline void order_clockwise(convex_polygon& points)
    {
        const size_t n = points.size();
        if (n < 3) return;
        double mx = 0.0, my = 0.0;
        for (const auto& p : points) { mx += p.x; my += p.y; }
        mx /= static_cast<double>(n);
        my /= static_cast<double>(n);
        std::sort(points.begin(), points.end(),
            [&](const Vector& p1, const Vector& p2)
            {
                return std::atan2(p1.y - my, p1.x - mx) < std::atan2(p2.y - my, p2.x - mx);
            });
    }

    inline double area(const convex_polygon& poly)
    {
        if (poly.size() < 3) return 0.0;
        double a = 0.0;
        size_t j = poly.size() - 1;
        for (size_t i = 0; i < poly.size(); ++i)
        {
            a += (poly[j].x + poly[i].x) * (poly[j].y - poly[i].y);
            j = i;
        }
        return std::fabs(a * 0.5);
    }

    inline convex_polygon get_intersection_poly(const convex_polygon& poly1, const convex_polygon& poly2)
    {
        if (poly1.size() < 3 || poly2.size() < 3) return {};
        convex_polygon clipped;
        clipped.reserve(poly1.size() + poly2.size() + 2 * poly1.size());

        for (const auto& p : poly1) if (inside_poly(p, poly2)) add_points(clipped, p);
        for (const auto& p : poly2) if (inside_poly(p, poly1)) add_points(clipped, p);
        for (size_t i = 0; i < poly1.size(); ++i)
        {
            size_t nxt = (i + 1) % poly1.size();
            add_points(clipped, get_intersections(poly1[i], poly1[nxt], poly2));
        }

        if (clipped.size() >= 3) order_clockwise(clipped); else clipped.clear();
        return clipped;
    }

    inline int ccw(const Vector& a, const Vector& b, const Vector& c)
    {
        double ar = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (is_equal(ar, 0.0)) return 0;
        return (ar > 0.0) ? -1 : 1;
    }

    inline void graham_scan(convex_polygon& points)
    {
        if (points.size() < 3) return;

        convex_polygon unique_pts;
        unique_pts.reserve(points.size());
        for (const auto& p : points) add_points(unique_pts, p);
        points.swap(unique_pts);
        if (points.size() < 3) return;

        size_t pivot_idx = 0;
        for (size_t i = 1; i < points.size(); ++i)
            if (points[i].y < points[pivot_idx].y ||
                (is_equal(points[i].y, points[pivot_idx].y) && points[i].x < points[pivot_idx].x))
                pivot_idx = i;

        std::swap(points[0], points[pivot_idx]);
        const Vector pivot = points[0];

        std::sort(points.begin() + 1, points.end(),
            [&](const Vector& a, const Vector& b)
            {
                int o = ccw(pivot, a, b);
                if (o == 0) return pivot.DistToSqr(a) < pivot.DistToSqr(b);
                return o == -1;
            });

        std::stack<Vector> st;
        st.push(points[0]);
        st.push(points[1]);

        for (size_t i = 2; i < points.size(); ++i)
        {
            while (st.size() >= 2)
            {
                Vector top = st.top(); st.pop();
                Vector nxt = st.top();
                if (ccw(nxt, top, points[i]) == -1)
                {
                    st.push(top);
                    break;
                }
            }
            st.push(points[i]);
        }

        points.resize(st.size());
        for (size_t i = st.size(); i-- > 0;)
        {
            points[i] = st.top();
            st.pop();
        }
    }
}
