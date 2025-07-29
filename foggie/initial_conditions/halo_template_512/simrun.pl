#! /usr/bin/perl
#-----------------------------------------------------------------------------
#
# Automated simulation runner.
#
# Copyright (c) 2012-2014, Britton Smith <brittonsmith@gmail.com>
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

use Cwd;
my $cdir = getcwd;
$job_name = join "/", (split "/", $cdir)[-2 .. -1];

$email_address = 'tumlinson@stsci.edu'; # put email in singe quotes
$job_file = "run_enzo.qsub";
$parameter_file = (glob("*.enzo"))[0];
$enzo_executable = "/nobackupnfs1/jtumlins/enzo-foggie-opthigh/src/enzo/enzo.exe"; 
$walltime = 360000;

while ($arg = shift @ARGV) {
    if ($arg =~ /^-mpi$/) {
	$mpi_command = shift @ARGV;
    }
    elsif ($arg =~ /^-wall/) {
	$walltime = shift @ARGV;
    }
    elsif ($arg =~ /^-pf/) {
	$parameter_file = shift @ARGV;
    }
    elsif ($arg =~ /^-exe/) {
	$enzo_executable = shift @ARGV;
    }
    elsif ($arg =~ /^-jf/) {
	$job_file = shift @ARGV;
    }
    elsif ($arg =~ /^-email/) {
        $email_address = shift @ARGV;
    }
    elsif ($arg =~ /^-h/) {
        &print_help();
    }
    else {
        &print_help();
    }
}

die "No mpi call given.\n" unless ($mpi_command);

$output_file = "estd.out";

$run_finished_file = "RunFinished";
$enzo_log_file = "OutputLog";
$log_file = "run.log";

$last_output = &get_last_output();
&change_parameters($last_output);
$first_output = $last_output;

$start_time = time;
while (1) {

    if ($last_output) {
	$run_par_file = $last_output;
	$enzo_flags = "-d -r";
    }
    else {
	$run_par_file = $parameter_file;
	$enzo_flags = "-d";
    }

    $command_line = "$mpi_command $enzo_executable $enzo_flags $run_par_file >& $output_file";
    print "Running: $command_line\n";
    &write_log("Starting enzo with $run_par_file.\n");
    $last_run_time = time;
    system($command_line);

    &change_parameters($last_output);
    $last_output = &get_last_output();

    if (($last_output eq $run_par_file) || !($last_output)) {
	&write_log("Simulation did not make new data, exiting.\n");
    &send_email("\'supercomputer says: $job_name in trouble!\'",
		"Hey,\nThe simulation exited without making new data.\nPlease help!\n");
	exit(0);
    }
    $directory = (split "/", $last_output)[1];
    $command_line = "shiftc --create-tar $directory lou:Halos/halo_002391/nref11n/$directory.tar";

    if (-e $run_finished_file) {
	&write_log("Simulation finished, exiting.\n");
    &send_email("\'supercomputer says: $job_name finished!\'",
		"Hey,\nDon\'t get too excited, but I think this simulation may be done!\n");
	exit(0);
    }
    if ($walltime) {
	$time_elapsed = time - $last_run_time;
	$time_left = $start_time + $walltime - time;
	if (1.1 * $time_elapsed > $time_left) {
	    &write_log("Insufficient time remaining to reach next output.\n");
	    $newid = &submit_job();
	    $last_output = &get_last_output();
	      &send_email("\'supercomputer says: $job_name stopped for today\'",
			  "Job started at: $first_output.\nJob ended at: $last_output.\nResubmitted as: $newid.\n");
	    exit(0);
	}
    }

}

sub write_log {
    my ($line) = @_;
    open (LOG, ">>$log_file");
    print LOG scalar (localtime);
    print LOG " $line";
    close (LOG);
}

sub get_last_output {
    open (IN, "<$enzo_log_file") or return;
    my @lines = <IN>;
    close (IN);

    my @online = split " ", $lines[-1];
    return $online[2];
}

sub send_email {
    my ($subject, $body) = @_;
    $signature = "-Robot Britton\n";
    $message_file = ".message";
    open (MAIL, ">$message_file") or die "Couldn't write message file.\n";
    print MAIL $email_address . "\n";
    print MAIL $subject . "\n";
    print MAIL $body;
    print MAIL $signature;
    close(MAIL);
}

sub submit_job {
    $jobid = `qsub $job_file`;
    chomp $jobid;
    return $jobid;
}

sub change_parameters {
    my ($parFile) = @_;
    $newParFile = $parFile . ".new";
    $oldParFile = $parFile . ".old";

    my $change_file = "new_pars";
    if (!(-e $change_file)) {
	return;
    }

    open (IN, "<$change_file") or return;
    my @lines = <IN>;
    close (IN);

    %newPars = ();
    foreach $line (@lines) {
	my ($my_key, $my_val) = split "=", $line, 2;
	$my_key =~ s/\s//g;
	$my_val =~ s/\s//g;
	$newPars{$my_key} = $my_val;
    }

    foreach $key (keys %newPars) {
	$changed{$key} = 0;
    }

    open (IN,"<$parFile") or die "Couldn't open $parFile.\n";
    open (OUT,">$newParFile") or die "Couldn't open $newParFile.\n";
    while (my $line = <IN>) {
	my $did = 0;
      PAR: foreach $par (keys %newPars) {
	  if ($line =~ /^\s*$par\s*=\s*/) {
	      &write_log("Switching $par to $newPars{$par}.\n");
	      print OUT "$par = $newPars{$par}\n";
	      $changed{$par} = 1;
	      $did = 1;
	      last PAR;
	  }
      }
	print OUT $line unless($did);
    }
    foreach $par (keys %changed) {
	unless ($changed{$par}) {
	    &write_log("Adding $par parameter set to $newPars{$par}.\n");
	    print OUT "$par = $newPars{$par}\n";
	}
    }
    close (IN);
    close (OUT);

    system ("mv $parFile $oldParFile");
    system ("mv $newParFile $parFile");
    my $new_change_file = $change_file . ".old";
    system ("mv $change_file $new_change_file");
}

sub print_help {
    print "Usage: $0 -mpi <mpi command> [options]\n";
    print "Options:\n";
    print "  -email <email address>\n";
    print "  -wall <walltime in seconds> Default: 86400 (24 hours)\n";
    print "  -pf <simulation parameter file> Default: *.enzo\n";
    print "  -exe <enzo executable> Default: enzo.exe\n";
    print "  -jf <job script> Default: run_enzo.qsub\n";
    exit(0);
}
