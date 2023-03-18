/** @file RNAup_cmdl.h
 *  @brief The header file for the command line option parser
 *  generated by GNU Gengetopt version 2.23
 *  http://www.gnu.org/software/gengetopt.
 *  DO NOT modify this file, since it can be overwritten
 *  @author GNU Gengetopt */

#ifndef RNAUP_CMDL_H
#define RNAUP_CMDL_H

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h> /* for FILE */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef RNAUP_CMDLINE_PARSER_PACKAGE
/** @brief the program name (used for printing errors) */
#define RNAUP_CMDLINE_PARSER_PACKAGE "RNAup"
#endif

#ifndef RNAUP_CMDLINE_PARSER_PACKAGE_NAME
/** @brief the complete program name (used for help and version) */
#define RNAUP_CMDLINE_PARSER_PACKAGE_NAME "RNAup"
#endif

#ifndef RNAUP_CMDLINE_PARSER_VERSION
/** @brief the program version */
#define RNAUP_CMDLINE_PARSER_VERSION VERSION
#endif

/** @brief Where the command line options are stored */
struct RNAup_args_info
{
  const char *help_help; /**< @brief Print help and exit help description.  */
  const char *detailed_help_help; /**< @brief Print help, including all details and hidden options, and exit help description.  */
  const char *full_help_help; /**< @brief Print help, including hidden options, and exit help description.  */
  const char *version_help; /**< @brief Print version and exit help description.  */
  int constraint_flag;	/**< @brief Apply structural constraint(s) during prediction. (default=off).  */
  const char *constraint_help; /**< @brief Apply structural constraint(s) during prediction. help description.  */
  int no_output_file_flag;	/**< @brief Do not produce an output file. (default=off).  */
  const char *no_output_file_help; /**< @brief Do not produce an output file. help description.  */
  int no_header_flag;	/**< @brief Do not produce a header with the command line parameters used in the outputfile. (default=off).  */
  const char *no_header_help; /**< @brief Do not produce a header with the command line parameters used in the outputfile. help description.  */
  int noconv_flag;	/**< @brief Do not automatically substitude nucleotide \"T\" with \"U\". (default=off).  */
  const char *noconv_help; /**< @brief Do not automatically substitude nucleotide \"T\" with \"U\". help description.  */
  char ** ulength_arg;	/**< @brief Specify the length of the unstructured region in the output. (default='4').  */
  char ** ulength_orig;	/**< @brief Specify the length of the unstructured region in the output. original value given at command line.  */
  unsigned int ulength_min; /**< @brief Specify the length of the unstructured region in the output.'s minimum occurreces */
  unsigned int ulength_max; /**< @brief Specify the length of the unstructured region in the output.'s maximum occurreces */
  const char *ulength_help; /**< @brief Specify the length of the unstructured region in the output. help description.  */
  char * contributions_arg;	/**< @brief Specify the contributions listed in the output. (default='S').  */
  char * contributions_orig;	/**< @brief Specify the contributions listed in the output. original value given at command line.  */
  const char *contributions_help; /**< @brief Specify the contributions listed in the output. help description.  */
  int window_arg;	/**< @brief Set the maximal length of the region of interaction. (default='25').  */
  char * window_orig;	/**< @brief Set the maximal length of the region of interaction. original value given at command line.  */
  const char *window_help; /**< @brief Set the maximal length of the region of interaction. help description.  */
  int include_both_flag;	/**< @brief Include the probability of unpaired regions in both (b) RNAs. (default=off).  */
  const char *include_both_help; /**< @brief Include the probability of unpaired regions in both (b) RNAs. help description.  */
  int extend5_arg;	/**< @brief Extend the region of interaction in the target to some residues on the 5' side..  */
  char * extend5_orig;	/**< @brief Extend the region of interaction in the target to some residues on the 5' side. original value given at command line.  */
  const char *extend5_help; /**< @brief Extend the region of interaction in the target to some residues on the 5' side. help description.  */
  int extend3_arg;	/**< @brief Extend the region of interaction in the target to some residues on the 3' side..  */
  char * extend3_orig;	/**< @brief Extend the region of interaction in the target to some residues on the 3' side. original value given at command line.  */
  const char *extend3_help; /**< @brief Extend the region of interaction in the target to some residues on the 3' side. help description.  */
  int interaction_pairwise_flag;	/**< @brief Activate pairwise interaction mode. (default=off).  */
  const char *interaction_pairwise_help; /**< @brief Activate pairwise interaction mode. help description.  */
  int interaction_first_flag;	/**< @brief Activate interaction mode using first sequence only. (default=off).  */
  const char *interaction_first_help; /**< @brief Activate interaction mode using first sequence only. help description.  */
  double pfScale_arg;	/**< @brief Set scaling factor for Boltzmann factors to prevent under/overflows..  */
  char * pfScale_orig;	/**< @brief Set scaling factor for Boltzmann factors to prevent under/overflows. original value given at command line.  */
  const char *pfScale_help; /**< @brief Set scaling factor for Boltzmann factors to prevent under/overflows. help description.  */
  double temp_arg;	/**< @brief Rescale energy parameters to a temperature in degrees centigrade. (default='37.0').  */
  char * temp_orig;	/**< @brief Rescale energy parameters to a temperature in degrees centigrade. original value given at command line.  */
  const char *temp_help; /**< @brief Rescale energy parameters to a temperature in degrees centigrade. help description.  */
  int noTetra_flag;	/**< @brief Do not include special tabulated stabilizing energies for tri-, tetra- and hexaloop hairpins. (default=off).  */
  const char *noTetra_help; /**< @brief Do not include special tabulated stabilizing energies for tri-, tetra- and hexaloop hairpins. help description.  */
  int dangles_arg;	/**< @brief Specify \"dangling end\" model for bases adjacent to helices in free ends and multi-loops. (default='2').  */
  char * dangles_orig;	/**< @brief Specify \"dangling end\" model for bases adjacent to helices in free ends and multi-loops. original value given at command line.  */
  const char *dangles_help; /**< @brief Specify \"dangling end\" model for bases adjacent to helices in free ends and multi-loops. help description.  */
  int noLP_flag;	/**< @brief Produce structures without lonely pairs (helices of length 1). (default=off).  */
  const char *noLP_help; /**< @brief Produce structures without lonely pairs (helices of length 1). help description.  */
  int noGU_flag;	/**< @brief Do not allow GU pairs. (default=off).  */
  const char *noGU_help; /**< @brief Do not allow GU pairs. help description.  */
  int noClosingGU_flag;	/**< @brief Do not allow GU pairs at the end of helices. (default=off).  */
  const char *noClosingGU_help; /**< @brief Do not allow GU pairs at the end of helices. help description.  */
  char * paramFile_arg;	/**< @brief Read energy parameters from paramfile, instead of using the default parameter set..  */
  char * paramFile_orig;	/**< @brief Read energy parameters from paramfile, instead of using the default parameter set. original value given at command line.  */
  const char *paramFile_help; /**< @brief Read energy parameters from paramfile, instead of using the default parameter set. help description.  */
  char * nsp_arg;	/**< @brief Allow other pairs in addition to the usual AU,GC,and GU pairs..  */
  char * nsp_orig;	/**< @brief Allow other pairs in addition to the usual AU,GC,and GU pairs. original value given at command line.  */
  const char *nsp_help; /**< @brief Allow other pairs in addition to the usual AU,GC,and GU pairs. help description.  */
  int energyModel_arg;	/**< @brief Set energy model..  */
  char * energyModel_orig;	/**< @brief Set energy model. original value given at command line.  */
  const char *energyModel_help; /**< @brief Set energy model. help description.  */
  
  unsigned int help_given ;	/**< @brief Whether help was given.  */
  unsigned int detailed_help_given ;	/**< @brief Whether detailed-help was given.  */
  unsigned int full_help_given ;	/**< @brief Whether full-help was given.  */
  unsigned int version_given ;	/**< @brief Whether version was given.  */
  unsigned int constraint_given ;	/**< @brief Whether constraint was given.  */
  unsigned int no_output_file_given ;	/**< @brief Whether no_output_file was given.  */
  unsigned int no_header_given ;	/**< @brief Whether no_header was given.  */
  unsigned int noconv_given ;	/**< @brief Whether noconv was given.  */
  unsigned int ulength_given ;	/**< @brief Whether ulength was given.  */
  unsigned int contributions_given ;	/**< @brief Whether contributions was given.  */
  unsigned int window_given ;	/**< @brief Whether window was given.  */
  unsigned int include_both_given ;	/**< @brief Whether include_both was given.  */
  unsigned int extend5_given ;	/**< @brief Whether extend5 was given.  */
  unsigned int extend3_given ;	/**< @brief Whether extend3 was given.  */
  unsigned int interaction_pairwise_given ;	/**< @brief Whether interaction_pairwise was given.  */
  unsigned int interaction_first_given ;	/**< @brief Whether interaction_first was given.  */
  unsigned int pfScale_given ;	/**< @brief Whether pfScale was given.  */
  unsigned int temp_given ;	/**< @brief Whether temp was given.  */
  unsigned int noTetra_given ;	/**< @brief Whether noTetra was given.  */
  unsigned int dangles_given ;	/**< @brief Whether dangles was given.  */
  unsigned int noLP_given ;	/**< @brief Whether noLP was given.  */
  unsigned int noGU_given ;	/**< @brief Whether noGU was given.  */
  unsigned int noClosingGU_given ;	/**< @brief Whether noClosingGU was given.  */
  unsigned int paramFile_given ;	/**< @brief Whether paramFile was given.  */
  unsigned int nsp_given ;	/**< @brief Whether nsp was given.  */
  unsigned int energyModel_given ;	/**< @brief Whether energyModel was given.  */

} ;

/** @brief The additional parameters to pass to parser functions */
struct RNAup_cmdline_parser_params
{
  int override; /**< @brief whether to override possibly already present options (default 0) */
  int initialize; /**< @brief whether to initialize the option structure RNAup_args_info (default 1) */
  int check_required; /**< @brief whether to check that all required options were provided (default 1) */
  int check_ambiguity; /**< @brief whether to check for options already specified in the option structure RNAup_args_info (default 0) */
  int print_errors; /**< @brief whether getopt_long should print an error message for a bad option (default 1) */
} ;

/** @brief the purpose string of the program */
extern const char *RNAup_args_info_purpose;
/** @brief the usage string of the program */
extern const char *RNAup_args_info_usage;
/** @brief the description string of the program */
extern const char *RNAup_args_info_description;
/** @brief all the lines making the help output */
extern const char *RNAup_args_info_help[];
/** @brief all the lines making the full help output (including hidden options) */
extern const char *RNAup_args_info_full_help[];
/** @brief all the lines making the detailed help output (including hidden options and details) */
extern const char *RNAup_args_info_detailed_help[];

/**
 * The command line parser
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int RNAup_cmdline_parser (int argc, char **argv,
  struct RNAup_args_info *args_info);

/**
 * The command line parser (version with additional parameters - deprecated)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param override whether to override possibly already present options
 * @param initialize whether to initialize the option structure my_args_info
 * @param check_required whether to check that all required options were provided
 * @return 0 if everything went fine, NON 0 if an error took place
 * @deprecated use RNAup_cmdline_parser_ext() instead
 */
int RNAup_cmdline_parser2 (int argc, char **argv,
  struct RNAup_args_info *args_info,
  int override, int initialize, int check_required);

/**
 * The command line parser (version with additional parameters)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param params additional parameters for the parser
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int RNAup_cmdline_parser_ext (int argc, char **argv,
  struct RNAup_args_info *args_info,
  struct RNAup_cmdline_parser_params *params);

/**
 * Save the contents of the option struct into an already open FILE stream.
 * @param outfile the stream where to dump options
 * @param args_info the option struct to dump
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int RNAup_cmdline_parser_dump(FILE *outfile,
  struct RNAup_args_info *args_info);

/**
 * Save the contents of the option struct into a (text) file.
 * This file can be read by the config file parser (if generated by gengetopt)
 * @param filename the file where to save
 * @param args_info the option struct to save
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int RNAup_cmdline_parser_file_save(const char *filename,
  struct RNAup_args_info *args_info);

/**
 * Print the help
 */
void RNAup_cmdline_parser_print_help(void);
/**
 * Print the full help (including hidden options)
 */
void RNAup_cmdline_parser_print_full_help(void);
/**
 * Print the detailed help (including hidden options and details)
 */
void RNAup_cmdline_parser_print_detailed_help(void);
/**
 * Print the version
 */
void RNAup_cmdline_parser_print_version(void);

/**
 * Initializes all the fields a RNAup_cmdline_parser_params structure 
 * to their default values
 * @param params the structure to initialize
 */
void RNAup_cmdline_parser_params_init(struct RNAup_cmdline_parser_params *params);

/**
 * Allocates dynamically a RNAup_cmdline_parser_params structure and initializes
 * all its fields to their default values
 * @return the created and initialized RNAup_cmdline_parser_params structure
 */
struct RNAup_cmdline_parser_params *RNAup_cmdline_parser_params_create(void);

/**
 * Initializes the passed RNAup_args_info structure's fields
 * (also set default values for options that have a default)
 * @param args_info the structure to initialize
 */
void RNAup_cmdline_parser_init (struct RNAup_args_info *args_info);
/**
 * Deallocates the string fields of the RNAup_args_info structure
 * (but does not deallocate the structure itself)
 * @param args_info the structure to deallocate
 */
void RNAup_cmdline_parser_free (struct RNAup_args_info *args_info);

/**
 * Checks that all the required options were specified
 * @param args_info the structure to check
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @return
 */
int RNAup_cmdline_parser_required (struct RNAup_args_info *args_info,
  const char *prog_name);


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* RNAUP_CMDL_H */
