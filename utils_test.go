package codectx

import (
	"reflect"
	"testing"
)

func TestUniqueElements(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]string
		expected []string
	}{
		{
			name: "single slice with filesystem paths",
			input: [][]string{
				{"/home/user/docs", "/home/user/images", "/home/user/docs"},
			},
			expected: []string{"/home/user/docs", "/home/user/images"},
		},
		{
			name: "multiple slices with overlapping filesystem paths",
			input: [][]string{
				{"/home/user/docs", "/home/user/images"},
				{"/home/user/images", "/home/user/music"},
			},
			expected: []string{"/home/user/docs", "/home/user/images", "/home/user/music"},
		},
		{
			name:     "empty input with filesystem paths",
			input:    [][]string{{}},
			expected: []string{},
		},
		{
			name: "completely unique filesystem paths",
			input: [][]string{
				{"/var/log/syslog", "/var/log/auth.log"},
				{"/etc/passwd", "/etc/hosts"},
			},
			expected: []string{"/var/log/syslog", "/var/log/auth.log", "/etc/passwd", "/etc/hosts"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := uniqueElements(test.input...)

			if !reflect.DeepEqual(got, test.expected) {
				t.Errorf("uniqueElementsOrdered(%v) = %v; want %v", test.input, got, test.expected)
			}
		})
	}
}

func TestComputeHash(t *testing.T) {
	tests := []struct {
		name     string
		input    []byte
		expected string
	}{
		{
			name:     "empty input",
			input:    []byte(""),
			expected: "d41d8cd98f00b204e9800998ecf8427e",
		},
		{
			name:     "simple string",
			input:    []byte("hello"),
			expected: "5d41402abc4b2a76b9719d911017c592",
		},
		{
			name:     "longer string",
			input:    []byte("The quick brown fox jumps over the lazy dog"),
			expected: "9e107d9d372bb6826bd81d3542a419d6",
		},
		{
			name:     "string with special characters",
			input:    []byte("!@#$%^&*()_+"),
			expected: "04dde9f462255fe14b5160bbf2acffe8",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := computeHash(test.input)
			if got != test.expected {
				t.Errorf("computeHash(%v) = %v; want %v", test.input, got, test.expected)
			}
		})
	}
}
