package crypto

import (
	"crypto/ed25519"
	"crypto/rand"
	"io"

	b58 "github.com/mr-tron/base58/base58"
)

type PrivKey ed25519.PrivateKey
type PubKey ed25519.PublicKey

const PublicKeySize = ed25519.PublicKeySize

func New() (PubKey, PrivKey) {
	pub, priv, _ := FromSeed(rand.Reader)
	return pub, priv
}

func FromSeed(src io.Reader) (PubKey, PrivKey, error) {
	pb, pr, err := ed25519.GenerateKey(src)
	if err != nil {
		return PubKey{}, PrivKey{}, err
	}

	var pub PubKey
	var priv PrivKey
	copy(pub[:], pb)
	copy(priv[:], pr)

	return pub, priv, nil
}

func FromReader(src io.Reader) (PubKey, PrivKey, error) {
	priv := make(PrivKey, ed25519.PrivateKeySize)
	pub := make(PubKey, ed25519.PublicKeySize)

	if _, err := src.Read(priv); err != nil {
		return nil, nil, err
	}
	if _, err := src.Read(pub); err != nil {
		return nil, nil, err
	}

	return pub, priv, nil
}

func (k PrivKey) String() string {
	return b58.Encode([]byte(k))
}

func (k PubKey) String() string {
	return b58.Encode([]byte(k))
}
