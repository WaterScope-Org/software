#!/usr/bin/expect -f

set user test

set password [exec cat password.txt]

puts $user

puts $password

spawn ./update_routine.sh

expect "Username"

send -- "sammy93\r"

expect "password"

send -- $password

send -- "\r"


set timeout 10

expect

expect eof
#!/bin/bash

