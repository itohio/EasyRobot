package eskf

import "github.com/chewxy/math32"

// covariance is the error state's covariance (δθ, δv, δp, δb_g, δb_a) and
// the scratch its propagation needs: propagation, scalar update, guard.
type covariance struct {
	p   [n][n]float32
	ap  [n][n]float32     // scratch: the nonzero rows of A·P
	apa [iBg][iBg]float32 // scratch: A·P·Aᵀ's rows and columns θ, v, p
}

// init sets the covariance to the diagonal d.
func (cv *covariance) init(d *[n]float32) {
	cv.p = [n][n]float32{}
	for i := range n {
		cv.p[i][i] = d[i]
	}
}

// grow adds v² to the variances of the three states from o.
func (cv *covariance) grow(o int, v float32) {
	for i := range 3 {
		cv.p[o+i][o+i] += v * v
	}
}

// sd is state i's standard deviation.
func (cv *covariance) sd(i int) float32 { return math32.Sqrt(cv.p[i][i]) }

// propagate: P ← Φ·P·Φᵀ + Q dt, Φ = I + A·dt, i.e. P += dt(AP + PAᵀ) +
// dt²·APAᵀ + Q dt, A's nonzero blocks A[θ,bg] = −R,
// A[v,θ] = −[a_w + g ẑ]×, A[v,ba] = −R, A[p,v] = I. The velocity and
// position variances are then inflated by a hair (×(1 + 1e-5) a sample,
// 0.1 %/s at 100 Hz): in long motion without an observation the yaw, velocity and
// position errors grow nearly collinear (the yaw error times the
// acceleration feeds the velocity, which the position integrates;
// correlations past 0.9998), a block float32 rounds to indefinite.
func (cv *covariance) propagate(r *[3][3]float32, aw [3]float32, cfg *Config, dt float32) {
	c := &cv.p
	ap := &cv.ap
	ax, ay, az := aw[0], aw[1], aw[2]+cfg.Gravity
	for j := range n {
		b0, b1, b2 := c[iBg][j], c[iBg+1][j], c[iBg+2][j]
		t0, t1, t2 := c[iTh][j], c[iTh+1][j], c[iTh+2][j]
		e0, e1, e2 := c[iBa][j], c[iBa+1][j], c[iBa+2][j]
		for i := range 3 {
			ap[iTh+i][j] = -(r[i][0]*b0 + r[i][1]*b1 + r[i][2]*b2)
		}
		ap[iV][j] = az*t1 - ay*t2 - (r[0][0]*e0 + r[0][1]*e1 + r[0][2]*e2)
		ap[iV+1][j] = -az*t0 + ax*t2 - (r[1][0]*e0 + r[1][1]*e1 + r[1][2]*e2)
		ap[iV+2][j] = ay*t0 - ax*t1 - (r[2][0]*e0 + r[2][1]*e1 + r[2][2]*e2)
		ap[iP][j], ap[iP+1][j], ap[iP+2][j] = c[iV][j], c[iV+1][j], c[iV+2][j]
	}
	// AP's rows are θ, v, p (0–8); add dt·AP and its transpose
	for i := range iBg {
		for j := range n {
			d := dt * ap[i][j]
			c[i][j] += d
			c[j][i] += d
		}
	}
	// and dt²·A·P·Aᵀ, so the step is Φ·P·Φᵀ with Φ = I + A·dt: to first
	// order alone it dropped that positive term, and P lost definiteness
	// (the correlation matrix's smallest eigenvalue reached −2.6)
	apa := &cv.apa
	for i := range iBg {
		w := &ap[i]
		b0, b1, b2 := w[iBg], w[iBg+1], w[iBg+2]
		t0, t1, t2 := w[iTh], w[iTh+1], w[iTh+2]
		e0, e1, e2 := w[iBa], w[iBa+1], w[iBa+2]
		for k := range 3 {
			apa[i][iTh+k] = -(r[k][0]*b0 + r[k][1]*b1 + r[k][2]*b2)
		}
		apa[i][iV] = az*t1 - ay*t2 - (r[0][0]*e0 + r[0][1]*e1 + r[0][2]*e2)
		apa[i][iV+1] = -az*t0 + ax*t2 - (r[1][0]*e0 + r[1][1]*e1 + r[1][2]*e2)
		apa[i][iV+2] = ay*t0 - ax*t1 - (r[2][0]*e0 + r[2][1]*e1 + r[2][2]*e2)
		apa[i][iP], apa[i][iP+1], apa[i][iP+2] = w[iV], w[iV+1], w[iV+2]
	}
	dt2 := dt * dt
	for i := range iBg {
		for j := i; j < iBg; j++ {
			d := dt2 * 0.5 * (apa[i][j] + apa[j][i])
			c[i][j] += d
			if j != i {
				c[j][i] += d
			}
		}
	}
	cg, ca := cfg.GyroNoise, cfg.AccelNoise
	wg, wa := cfg.GyroBiasWalk, cfg.AccelBiasWalk
	for i := range 3 {
		c[iTh+i][iTh+i] += cg * cg * dt
		c[iV+i][iV+i] += ca * ca * dt
		c[iBg+i][iBg+i] += wg * wg * dt
		c[iBa+i][iBa+i] += wa * wa * dt
	}
	for i := iV; i < iP+3; i++ {
		c[i][i] *= 1 + 1e-5
	}
}

// row is a measurement's row of H: m ≤ 4 nonzeros val at idx.
type row struct {
	idx [4]int
	val [4]float32
	m   int
}

// update is one scalar measurement y = h·δx + noise of variance rv: it
// accumulates the correction into dx and downdates the covariance. The
// states in consider (a bit per state) are consider states (Schmidt): the
// measurement doesn't correct them, only keeps their correlations right;
// that gain isn't the optimal one, so the covariance takes the Joseph
// form, P − K·hP − (K·hP)ᵀ + K·s·Kᵀ.
func (cv *covariance) update(h *row, y, rv float32, consider uint16, dx *[n]float32) {
	c := &cv.p
	var ph [n]float32
	for i := range n {
		var s float32
		for k := range h.m {
			s += c[i][h.idx[k]] * h.val[k]
		}
		ph[i] = s
	}
	s := rv
	var hd float32
	for k := range h.m {
		s += h.val[k] * ph[h.idx[k]]
		hd += h.val[k] * dx[h.idx[k]]
	}
	e := y - hd
	inv := 1 / s
	var k [n]float32
	for i := range n {
		if consider&(1<<i) != 0 {
			continue
		}
		k[i] = ph[i] * inv
		dx[i] += k[i] * e
	}
	for i := range n {
		for j := i; j < n; j++ {
			v := c[i][j] - k[i]*ph[j] - k[j]*ph[i] + k[i]*k[j]*s
			c[i][j] = v
			c[j][i] = v
		}
	}
}

// guard keeps the covariance a covariance in float32 (after each batch of
// scalar updates): a positive diagonal and correlations within ±1 (P −
// k·phᵀ cancels to rounding when a measurement is far surer than the
// state, and a negative variance would turn the next gain around).
func (cv *covariance) guard() {
	c := &cv.p
	var sd [n]float32
	for i := range n {
		c[i][i] = max(c[i][i], 1e-12)
		sd[i] = math32.Sqrt(c[i][i])
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lim := sd[i] * sd[j]
			v := min(max(c[i][j], -lim), lim)
			c[i][j], c[j][i] = v, v
		}
	}
}
