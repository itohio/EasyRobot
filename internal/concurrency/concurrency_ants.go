// +build ants

package concurrency

func Submit(f func()) {
	// FIXME
	go f()
}
