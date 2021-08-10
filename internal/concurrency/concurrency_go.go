// +build !ants

package concurrency

func Submit(f func()) {
	go f()
}
