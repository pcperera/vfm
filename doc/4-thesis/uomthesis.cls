% $Log: uomthesis.cls,v $
% Revision 1.0 25/11/2021 Nisansa de Silva

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Based on 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mitthesis.cls,v $
% Revision 1.9  2012/02/15 15:48:16  jdreed
% Tweak the "permission" statement per the Libraries' specs
% retrieved 15-Feb-2012
%
% Revision 1.8  2009/07/27 14:23:50  boojum
% added changing draft mark
%
% Revision 1.6  2005/05/13 19:57:40  boojum
% added leftblank option
%
% Revision 1.5  2002/04/18 14:10:08  boojum
% changed doublespace into setspace for 2e compliance
%
% Revision 1.4  2001/02/08 18:57:13  boojum
% turned two newpages into cleardoublepages
%
% Revision 1.3  2000/08/14 14:53:29  boojum
% commented out footheight, which is relevant for drafthead
%
% Revision 1.2  1999/10/21 14:51:33  boojum
% changed references to documentstyle to documentclass in comments
%
% Revision 1.1  1999/10/21 14:39:31  boojum
% Initial revision
%
%Revision 1.7  1998/04/01  20:45:34  othomas
%removed offending phrase ", and to grant others the right to do so" from copyright notice.
%
%Revision 1.6  96/06/26  15:07:29  othomas
%safety checkin.
%
%Revision 1.5  93/06/07  15:38:50  starflt
%Altered 'vi' option copyright wording to comply with new Institute
%Archives requirements and MIT lawyers.
%
%Revision 1.4  92/08/19  16:51:06  lwvanels
%Updated Course 6 title page for new permissions.
%
%Revision 1.3  92/04/23  10:16:15  epeisach
%Fixed comment character in rcs file
%
%Revision 1.2  92/04/22  13:12:02  epeisach
%Fixes for 1991 course 6 requirements
%Phrase "and to grant others the right to do so" has been added to 
%permission clause
%Second copy of abstract is not counted as separate pages so numbering works
%out
%
%Revision 1.1  90/05/04  11:45:53  lwvanels
%Initial revision

%
% LaTeX format for theses at MIT
% Based on "Specifications for Thesis Preparation" 

% `vi' and `upcase' options by Krishna Sethuraman - krishna@athena.mit.edu
% Margins and heading types by Peter Nuth  - nuth@ai.mit.edu
% Title and abstract page by Stephen Gildea - gildea@erl.mit.edu
% Look in this directory for example file mitthesis.doc
% Also for propcover.tex - Boilerplate for PHD proposal.

% To use this style - say something like:
%  for dull, boring thesis format:
%	\documentclass[12pt]{mitthesis}
%       \pagestyle{plain}
% OR for fast drafts: 
%	\documentclass[11pt,singlespace,draft]{mitthesis}
%	\pagestyle{drafthead}
% OR for Tech Reports:
%	\documentclass[12pt,twoside]{mitthesis}	
%	\pagestyle{headings}
% OR
%  some other combination...
%
%%%% New options:
% 
% Option `twoside':
%   Good for producing Tech Reports.
%   The default is single-sided printing, which is what M.I.T. wants on the
%   thesis document itself.
%
% Option `singlespace':
%   Good for drafts.
%   Double-spaced theses are the default.
%   That is what M.I.T. asks for in the formal specifications.
%
% 	Note that MIT does not REQUIRE all theses to be double-spaced anymore.
% 	Someone in the library system said that it's OK to be single-spaced.
% 	(Regardless of what the specs. say...)
%   To get singlespacing in an area - Use  the 'singlespace' environment. 
%
% Option `draft':
%   Puts `overfull' boxes at the end of lines that are too long. 
%
% Pagestyle `drafthead':
%   Puts the date and the label ``*DRAFT*'' in the footer.
%
%%%%%%%%%%
%
%%%% Parameters to initialize for boilerplate page:
%
%	\title{Mixed Circular Cylindrical Shells}
% 	\author{J. Casey Salas}
% 	\prevdegrees{B.S., University of California (1978) \\
%		     S.M., Massachusetts Institute of Technology (1981)}
% 	\department{Department of Electrical Engineering and Computer Science}
% 	\degree{Doctor of Philosophy}
%% If the thesis is for two degrees simultaneously, list them both
%% separated by \and like this:
% 	\degree{Doctor of Philosophy \and Master of Science}
% 	\degreemonth{February}
% 	\degreeyear{1987}
% 	\thesisdate{December 10, 1986}
%% If the thesis is copyright by the Institute, leave this line out and
%% the standard copyright line will be used instead.
% 	\copyrightnotice{J. Casey Salas, 1986}
%% If there is more than one supervisor, use the \supervisor command
%% once for each.
% 	\supervisor{John D. Galli}{Director, Sound Instrument Laboratory}
%% This is the department committee chairman, not the thesis committee chairman
% 	\chairman{Arthur C. Smith}
%		 {Chairman, Departmental Committee on Graduate Students}
%% Make the titlepage based on the above information.  If you need
%% something special and can't use the standard form, you can specify
%% the exact text of the titlepage yourself.  Put it in a titlepage
%% environment and leave blank lines where you want vertical space.
%% The spaces will be adjusted to fill the entire page.  The dotted
%% lines for the signatures are made with the \signature command.
%
%% The abstractpage environment sets up everything on the page except
%% the text itself.  The title and other header material are put at the
%% top of the page, and the supervisors are listed at the bottom.  A
%% new page is begun both before and after.  Of course, an abstract may
%% be more than one page itself.  If you need more control over the
%% format of the page, you can use the abstract environment, which puts
%% the word "Abstract" at the beginning and single spaces its text.
%
% 	\begin{abstractpage}
%	    Abstract goes here.
%	\end{abstractpage}
%
%%%%%%%% Newer additions 
%
% documentclass options - 
% vi		For MIT course VI or VIII thesis - will copyright the thesis to
% 		you while giving MIT permission to copy and distribute it.
% upcase	Will put much of the cover page in uppercase, as per the
% 		example on page 17 of the *Specifications for Thesis
% 		Preparation*, (revised 1989)
% Also added ``All Rights Reserved'' to default copyright notice.
%
%%%%%%%%%%%
% 
% Documentclass options (vi and upcase) and changes to copyright notice
%	Copyright (c) 1990, by Krishna Sethuraman.
%
% Pagestyle and header generation
%	Copyright (c) 1987, 1988 by Peter Nuth
%
% Original version
%	 Copyright (c) 1987 by Stephen Gildea
% Permission to copy all or part of this work is granted, provided
% that the copies are not made or distributed for resale, and that
% the copyright notice and this notice are retained.
% 
% THIS WORK IS PROVIDED ON AN "AS IS" BASIS.  THE AUTHOR PROVIDES NO
% WARRANTY WHATSOEVER, EITHER EXPRESS OR IMPLIED, REGARDING THE WORK,
% INCLUDING WARRANTIES WITH RESPECT TO ITS MERCHANTABILITY OR FITNESS
% FOR ANY PARTICULAR PURPOSE.
%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uomthesis}[25/11/2021]

\DeclareOption{doublespace}{}	% This is default
				% So we do not read this style twice
\DeclareOption{singlespace}{		% If they explicitly wants single spacing
    \typeout{Single spaced}
    \def\mystretch{1}}	

%% `vi' and `upcase' document style options.  Krishna Sethuraman (1990)
\newcount\vithesis
\DeclareOption{vi}{\typeout{Course VI/VIII thesis style.}\advance\vithesis by1}
\vithesis=0

\DeclareOption{upcase}{\typeout{Uppercase cover page.}
	\gdef\choosecase#1{\uppercase\expandafter{#1}}}
\def\choosecase#1{#1}

% Nisansa added to comply with UoM's dual IEEE and APA policy
\newif\if@APA \@APAfalse
\newif\ifapaT \apaTfalse
\DeclareOption{APA}{\@APAtrue \apaTtrue}

\newif\if@UoMdraft \@UoMdraftfalse
\DeclareOption{UoMdraft}{\@UoMdrafttrue}


% IEEE uses Times Roman font, so we'll default to Times.
% These three commands make up the entire times.sty package.
\renewcommand{\sfdefault}{phv}
\renewcommand{\rmdefault}{ptm}
\renewcommand{\ttdefault}{pcr}

\normalfont\selectfont


%% leftblank option by Kevin Fu
\newif\if@leftblank \@leftblankfalse

\DeclareOption{leftblank}{\typeout{Intentionally Leaving Pages Blank}
\@leftblanktrue}

%  Thesis looks much like report
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass{report}

% If the user wants single spacing, set baselinestretch=1.

\usepackage{setspace}
\usepackage[a4paper,total={170mm,257mm},top=25mm,includehead,right=25mm,left=40mm,bottom=25mm]{geometry}
%\usepackage[a4paper,total={210mm,297mm},top=15mm,includeheadfoot,right=25mm,left=40mm,bottom=40mm,headsep=5.5mm,footskip=10mm]{geometry}
 



%%%%%%%  Set up margins and formatting params %%%

%% Margins.
%%  Note we want 1in top margin assuming no header line, so push header
%%	into 1in margin.
%%  Draft mode brings the header back down.
%
%\setlength{\oddsidemargin}{0.25in}	% 1.25in left margin 
%\setlength{\evensidemargin}{0.25in}	% 1.25in left margin (even pages)
%\setlength{\topmargin}{0.0in}		% 1in top margin
%
%\setlength{\textwidth}{6.0in}		% 6.0in text - 1.25in rt margin
%\setlength{\textheight}{9in}		% Body ht for 1in margins
%\addtolength{\topmargin}{-\headheight}	% No header, so compensate
%\addtolength{\topmargin}{-\headsep}	% for header height and separation
%
%% The next two macros compensate page style for headers and footers
%% We only need them in page styles that USE headers and footers.
%    % If we have a header, it must be 1in from top of page.
%\def\pulldownheader{			% Shift header down 1in from top
%    \addtolength{\topmargin}{\headheight}	
%    \addtolength{\topmargin}{\headsep}	
%    \addtolength{\textheight}{-\headheight}
%    \addtolength{\textheight}{-\headsep}
%}
%    % If we have a footer, put it 1in up from bottom
%\def\pullupfooter{				% Shift footer up
%    \addtolength{\textheight}{-\footskip}
%%    \addtolength{\textheight}{-\footheight}  %footheight doesn't
%%    						exist in 2e
%}

%%%%%%%  End of margins and formatting params %%%

%%%%%%%  Fix various header and footer problems %%%

% Draft mark on the right side of left pages (outside)
% this mark is also the only one visible on single sided.
\newcommand{\draftrmark}{**DRAFT**} 
% Draft mark on the left side of right pages (outside)
\newcommand{\draftlmark}{**DRAFT**} % 

% Macros to make changing the Draft easier
\newcommand{\drmark}[1]{\renewcommand{\draftrmark}{#1}}
\newcommand{\dlmark}[1]{\renewcommand{\draftlmark}{#1}}
\newcommand{\dmark}[1]{\drmark{#1}\dlmark{#1}}


\if@UoMdraft  
    \def\mystretch{1.3}   % one-and-a-half spacing
\else 
    \def\mystretch{1.15} % Final Submission Spacing of UoM
\fi

% Note - doublespace.sty has some float-related troubles in
% combination with graphics or color, and is not officially compliant
% with 2e.  setspace is a replacement which is 2e-compliant.

% Read the doublespace style that we got from Rochester:
%\input setdoublespace.sty 		

\def\baselinestretch{\mystretch}	% Double spacing hack



% Format for draft of thesis.  Define our own PageStyle -
% Just like headings, but has foot lines with the date and warning

\if@twoside         % If two-sided printing.
\def\ps@drafthead{
    \let\@mkboth\markboth
    \def\@oddfoot{\rm \today \hfil \sc \draftrmark}
    \def\@evenfoot{\sc \draftlmark \hfil \rm \today }
    \def\@evenhead{\rm \thepage\hfil \sl \leftmark}
    \def\@oddhead{\hbox{}\sl \rightmark \hfil \rm\thepage}
    \def\chaptermark##1{\markboth {\uppercase{\ifnum \c@secnumdepth >\m@ne
	\@chapapp\ \thechapter. \ \fi ##1}}{}}
    \def\sectionmark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\z@
	\thesection. \ \fi ##1}}}
    \pulldownheader				% Bring header down from edge
    \pullupfooter				% Bring footer up
}
\else               % If one-sided printing.
\def\ps@drafthead{
    \let\@mkboth\markboth
    \def\@oddfoot{\rm \today \hfil \sc \draftrmark}
    \def\@oddhead{\hbox{}\sl \rightmark \hfil \rm\thepage}
    \def\chaptermark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\m@ne
	\@chapapp\ \thechapter. \ \fi ##1}}}
%    \pulldownheader				% Bring header down from edge
%    \pullupfooter				% Bring footer up
}
\fi

% I redefine these formats that were defined in report.sty
% Definition of 'headings' page style 
%  Note the use of ##1 for parameter of \def\chaptermark inside the
%  \def\ps@headings.
%

\if@twoside					% If two-sided printing.
\def\ps@headings{\let\@mkboth\markboth
    \def\@oddfoot{}
    \def\@evenfoot{}		% No feet.
    \def\@evenhead{\rm \thepage\hfil \sl \leftmark}	% Left heading.
    \def\@oddhead{\hbox{}\sl \rightmark \hfil \rm\thepage}	% Right heading.
    \def\chaptermark##1{\markboth {\uppercase{\ifnum \c@secnumdepth >\m@ne
	\@chapapp\ \thechapter. \ \fi ##1}}{}}	
    \def\sectionmark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\z@
	\thesection. \ \fi ##1}}}
    \pulldownheader				% Bring header down from edge
}
\else						% If one-sided printing.
\def\ps@headings{\let\@mkboth\markboth
    \def\@oddfoot{}
    \def\@evenfoot{}		%     No feet.
    \def\@oddhead{\hbox {}\sl \rightmark \hfil \rm\thepage}	% Heading.
    \def\chaptermark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\m@ne
	\@chapapp\ \thechapter. \ \fi ##1}}}
    \pulldownheader				% Bring header down from edge
}
\fi






% Redefinition of 'myheadings' page style.  
%
\def\ps@myheadings{\let\@mkboth\@gobbletwo
    \def\@oddfoot{}
    \def\@evenfoot{}
    \def\sectionmark##1{}
    \def\subsectionmark##1{}
    \def\@evenhead{\rm \thepage\hfil\sl\leftmark\hbox {}}	%
    \def\@oddhead{\hbox{}\sl\rightmark \hfil \rm\thepage}	%
    \pulldownheader				% Bring header down from edge
}						

% Redefine '/chapter' to always start on an odd page.
% Should make no difference in singleside mode.
%
\if@leftblank
% Prints "THIS PAGE INTENTIONALLY LEFT BLANK" on blank pages.
\def\chapter{\clearpage\ifodd\c@page\else
   \hbox{}\par\vfill\centerline%
   {THIS PAGE INTENTIONALLY LEFT BLANK}%
   \vfill\newpage\fi
   \thispagestyle{plain}	% Page style of chapter page is 'plain'
   \global\@topnum\z@		% Prevents figures from going at top of page.
   \@afterindentfalse		% Suppresses indent in first paragraph.  Change
   \secdef\@chapter\@schapter}	% to \@afterindenttrue to have indent.
\else
\def\chapter{\cleardoublepage	% Starts new page.
   \thispagestyle{plain}	% Page style of chapter page is 'plain'
   \global\@topnum\z@		% Prevents figures from going at top of page.
   \@afterindentfalse		% Suppresses indent in first paragraph.  Change
   \secdef\@chapter\@schapter}	% to \@afterindenttrue to have indent.
\fi
% If using the report style, use - instead of . in the figure number.
\@ifundefined{thechapter}{}{\def\thefigure{\thechapter-\arabic{figure}}}



\if@APA
    \usepackage[sectionbib]{natbib}
    %\usepackage[natbibapa]{apacite}
    \usepackage[labelsep=newline,singlelinecheck=false]{caption}
\else
    \usepackage[numbers,sort&compress,sectionbib]{natbib}
    \usepackage[center]{caption}
    \renewcommand{\figurename}{Fig.}
    \renewcommand{\tablename}{\textbf{TABLE}}
\fi




%%%%%%%%%  End of Style parameters %%%%

% Here's Gildea's Boilerplate Stuff.
% Copyright (c) 1987 by Stephen Gildea
% Permission to copy all or part of this work is granted, provided
% that the copies are not made or distributed for resale, and that
% the copyright notice and this notice are retained.

%% Define all the pieces that go on the title page and the abstract.

% \title and \author already exist
\def\department#1{\gdef\@department{#1}}
\def\keywords#1{\gdef\@keywords{#1}}
\def\indexN#1{\gdef\@indexN{#1}}

\newbox\@supervisorbox
\newcounter{supervisorCount}
\def\supervisorName#1{
\setbox\@supervisorbox\vbox{\unvbox\@supervisorbox\vspace{3em}\noindent Name of Supervisor: #1 \vskip 10pt\noindent Signature of the Supervisor: \hspace{12.3em} Date:
  }\refstepcounter{supervisorCount}
}

\newsavebox\@studentbox
\newbox\@studentSignbox
\newcounter{studentCount}
\def\student#1#2{
\setbox\@studentbox\vbox{\unvbox\@studentbox{\hfil\fontsize{14pt}{1.5em}\selectfont\choosecase{#2 (#1)}\hfil\vspace{2em}}}\setbox\@studentSignbox\vbox{\unvbox\@studentSignbox\vspace{2em}
%{\ifnum\thestudentCount>1 \noindent Name of Student: #2 \vskip 10pt\fi}
\noindent Signature: \hspace{20em} Date:}\gdef\@indexN{#1}\gdef\@author{#2}\refstepcounter{studentCount}}

\def\type#1{\gdef\@type{#1}}
\def\faculty#1{\gdef\@faculty{#1}}

\def\degree#1{\setbox0\hbox{#1}	\gdef\@degree{#1}} % If you are getting two degrees, use \and between the names.

% typically just a month and year
\def\degreemonth#1{\gdef\@degreemonth{#1}}
\def\degreeyear#1{\gdef\@degreeyear{#1}}

\def\thesisdate#1{\gdef\@thesisdate{#1}}
\thesisdate{\@degreemonth{}~\@degreeyear}



\newcommand{\studentString}{\ifnum\thestudentCount>1 \usebox\@studentbox \else

{\fontsize{14pt}{1.5em}\selectfont  \choosecase{\@author}}

\vspace{0.01\textheight}

{\fontsize{14pt}{1.5em}\selectfont  \choosecase{\@indexN}}

\fi}


%% `upcase' documentclass option: \choosecase is defined either as a dummy or
%% a macro to change the (expanded) argument to uppercase.
\newcommand{\setUpPage}[2]{
\newgeometry{left=40mm,right=40mm,top=40mm,bottom=0cm}
\setstretch{1} % Keep Constant
\begin{center}
 %   \cleardoublepage\centering
\thispagestyle{empty}
 
\large
{\fontsize{16pt}{1.5em}\selectfont\bf \MakeUppercase{\@title} \par}

\vspace{0.1\textheight}


\studentString

\vspace{0.1\textheight}

{\fontsize{#2}{1.5em}\selectfont \choosecase{#1} }

\vspace{0.1\textheight}

{\fontsize{14pt}{1.5em}\selectfont \choosecase{\@department \\ \@faculty}}

\vspace{0.05\textheight}

{\fontsize{14pt}{1.5em}\selectfont
    \choosecase{University of Moratuwa \\ Sri Lanka}
}

\vspace{0.05\textheight}

{\fontsize{14pt}{1.5em}\selectfont
    \choosecase{\@degreemonth{} \@degreeyear}
}
\end{center}
\restoregeometry
}



\def\maketitle{
% Cover Page
\begin{titlepage}
\setUpPage{\@degree}{14pt}
\end{titlepage}
\clearpage
% Title Page
\begin{titlepage}
\setUpPage{
\@type{} submitted in partial fulfillment of the requirements for the degree\\ \@degree{}
}{12pt}
\end{titlepage}

\newcommand{\supervisorString}{\ifnum\thesupervisorCount>1 supervisors \else supervisor\fi}
\newcommand{\supervisorPosessive}{\ifnum\thesupervisorCount>1 our \else my\fi}
\newcommand{\supervisorNoun}{\ifnum\thesupervisorCount>1 We \else I \fi}


\newcommand{\studentPosessive}{\ifnum\thestudentCount>1 our \else my\fi}
\newcommand{\studentNoun}{\ifnum\thestudentCount>1 We \else I \fi}




% Declaration Page

\pagestyle{plain}
\clearpage
\pagenumbering{roman} 

\chapter*{{\raggedright DECLARATION}\hfill}
\addcontentsline{toc}{chapter}{Declaration of the Candidate \& Supervisor}

\studentNoun{} declare that this is \studentPosessive{} own work and this \@type{} does not incorporate without acknowledgement any material previously submitted for a Degree or Diploma in any other University or Institute of higher learning and to the best of \studentPosessive{} knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text. \studentNoun{} retain the right to use this content in whole or part in future works (such as articles or books).

%\vspace{2em}
%\noindent Signature: \hspace{20em} Date:
\unvbox\@studentSignbox
\vspace{2em}

\noindent The \supervisorString{} should certify the \@type{} with the following declaration.

\vspace{2em}

\noindent The above candidate has carried out research for the \@degree{} \@type{} under \supervisorPosessive{} supervision. \supervisorNoun{} confirm that the declaration made above by the student is true and correct.  

\unvbox\@supervisorbox
}


% this environment should probably be called abstract,
% but we want people to also be able to get at the more
% basic abstract environment
\def\abstractpage{\cleardoublepage
\begin{abstract}\if@UoMdraft  
    \setstretch{1.6} % double spacing
\fi}

%% Changed from \unvbox to \unvcopy for use with multiple copies of abstract page.
%% Krishna Sethuraman (1990)
\def\endabstractpage{\end{abstract}\vspace{1cm}\noindent{\fontsize{11pt}{1em}\selectfont 
\noindent\textbf{Keywords}: \@keywords
}\newpage}

%% This counter is used to save the page number for the second copy of
%% the abstract.
\newcounter{savepage}

% You can use the titlepage environment to do it all yourself if you
% don't want to use \maketitle.  If the titlepage environment, the
% paragraph skip is infinitely stretchable, so if you leave a blank line
% between lines that you want space between, the space will stretch so
% that the title page fills up the entire page.
\def\titlepage{\cleardoublepage\centering
  \thispagestyle{empty}
  \parindent 0pt \parskip 10pt plus 1fil minus 1fil
  \def\baselinestretch{1}\@normalsize\vbox to \vsize\bgroup\vbox to 9in\bgroup}
% The \kern0pt pushes any depth into the height.  Thanks to Richard Stone.
\def\endtitlepage{\par\kern 0pt\egroup\vss\egroup\newpage}

\def\abstract{\unNumChapter{Abstract}\small\def\baselinestretch{1}\@normalsize}
\def\endabstract{\par}


% Create a Table that can switch between IEEE and APA styles
\newcounter{tableCount}
\def\uomTable#1#2#3#4{
    \if@APA  %APA
        \caption[#1]{\textit{#1}}
    \else % IEEE
        \centering
        \caption[#1]{\uppercase{#1}}
    \fi
    \label{#2}
    \centering
    \if@UoMdraft  
        \setstretch{1.6} % double spacing
    \fi
    #3
    \if\relax\detokenize{#4}\relax\else \\ \vspace{3mm}{{\raggedright \textit{Note.} #4}\hfill}\fi
    \refstepcounter{tableCount}
}

% Create a Figure that can switch between IEEE and APA styles
\newcounter{figureCount}
\def\uomFig#1#2#3#4{
    \if@APA  %APA
        \caption[#1]{\textit{#1}}
        \label{#2}
    \fi
    \centering
    #3
    \if@APA\else % IEEE
        \caption[#1]{#1}
        \label{#2}
    \fi
    \if\relax\detokenize{#4}\relax\else\if@APA \\ \fi\vspace{3mm}{{\raggedright \textit{Note.} #4}\hfill}\fi
    \refstepcounter{figureCount}
}


% This sets up the Chapter, Section, Subsection titles in the UoM format
\usepackage{titlesec}
\setcounter{secnumdepth}{4}
\renewcommand{\chaptertitlename}{CHAPTER}
\titleformat{\chapter}[display]
{\centering\fontsize{14}{1.5em}\selectfont\bfseries}{\chaptertitlename\ \thechapter}{4pt}{\fontsize{14}{1.5em}\selectfont\MakeUppercase}

\titleformat{name=\chapter,numberless}[display]
{\fontsize{14}{1.5em}\selectfont\bfseries}{}{0pt}{\fontsize{14}{1.5em}\selectfont\MakeUppercase}
\titlespacing*{\chapter}{0pt}{-20pt}{20pt}
\titleformat{\section}
  {\normalfont\fontsize{14}{15}\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}
  {\normalfont\fontsize{12}{15}\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\normalfont\fontsize{12}{15}\bfseries}{\thesubsubsection}{1em}{}



\usepackage[titletoc]{appendix}

\usepackage{tocloft}
\renewcommand*{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand\cftfigfont{\small}
\setlength{\cftfigindent}{0pt}
\renewcommand{\cftfigpresnum}{Figure }
\renewcommand{\cftfignumwidth}{6em}
\renewcommand{\cftfigafterpnum}{}

\renewcommand\cfttabfont{\small}
\setlength{\cfttabindent}{0pt}
\renewcommand{\cfttabpresnum}{Table }
\renewcommand{\cfttabnumwidth}{6em}
\renewcommand{\cfttabafterpnum}{}

\renewcommand{\cftbeforefigskip}{0em}
\renewcommand{\cftbeforechapskip}{0em}

\renewcommand{\contentsname}{{\fontsize{16pt}{1.5em}\selectfont TABLE OF CONTENTS}}

\renewcommand\cftchapfont{\mdseries}
\renewcommand\cftchappagefont{\mdseries}

\renewcommand\cftsecfont{\normalfont}
\renewcommand\cftsecpagefont{\normalfont}

\renewcommand{\cftdot}{}


\makeatletter
\def\@chapter[#1]#2{
\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
    \typeout{\@chapapp\space\thechapter.}%
    \addcontentsline{toc}{chapter}%
    {\protect\numberline{\thechapter}#1}%
\else
    \addcontentsline{toc}{chapter}{#1}%
    \fi
        \chaptermark{#1}%
        \if@twocolumn
           \@topnewpage[\@makechapterhead{\hfill{\centering #2}\hfill}]%
        \else
            \@makechapterhead{\hfill{\centering #2}\hfill}%
            \@afterheading
     \fi}
\makeatother

\newcommand\unNumChapter[1]{%
  \chapter*{#1}%
  \addcontentsline{toc}{chapter}{#1}}

% Set up Appendices
\newcounter{appenCount}
\newcommand{\listappendicesname}{Appendices}
\newlistof{appendices}{apc}{\listappendicesname}
\newcommand{\appendiceC}[2]{
\renewcommand{\chaptertitlename}{APPENDIX}
\chapter{#1}
\label{#2}
\addcontentsline{apc}{appendices}{Appendix -\Alph{chapter}\hspace{10mm}#1}\refstepcounter{appenCount}
}

% Create an abbreviation
\usepackage[acronym,nonumberlist,shortcuts,toc,nogroupskip]{glossaries}
\usepackage{xparse}

\setglossarystyle{long}
\renewcommand*{\acronymname}{List of Abbreviations} % rename title
\renewcommand*{\glspostdescription}{} % remove dot at the end of description
\renewcommand*{\glsclearpage}{} % Remove the clear page
\setglossarystyle{long3colheader}
\renewcommand*{\entryname}{Abbreviation}
%\renewcommand*{\descriptionname}{Description}
\renewcommand*{\pagelistname}{\newline}
\makeglossaries

\newcounter{abbrCount}
\makeatletter
\newcommand{\abbr}[3][\@nil]{%
  \def\tmp{#1}%
   \ifx\tmp\@nnil
      \newacronym{#3}{#3}{#2}
    \else
      \newacronym{#3}{#1}{#2}
    \fi
    #2 (\acs{#3})
    \refstepcounter{abbrCount}}
\makeatother





% Set up Footnotes
\newcommand{\footURL}[1]{\footnote{\url{#1}}}
\newcommand{\footReText}[2]{\footnote{\label{#2}#1}}
\newcommand{\footReURL}[2]{\footnote{\label{#2}\url{#1}}}
\makeatletter
\newcommand\footnoteref[1]{\protected@xdef\@thefnmark{\ref{#1}}\@footnotemark}
\makeatother

% Quotes
\usepackage{csquotes}
\usepackage{setspace}

\newenvironment{indentedQuote}[1][]{
    \begin{displayquote}
    \setstretch{1.6} % double spacing
}{
   \end{displayquote}
}

\newcommand{\latexC}[1]{\texttt{\textbackslash #1}}
  

\newcommand{\UoM}{University of Moratuwa}
\newcommand{\TheClass}{\texttt{uomthesis} class}


\newcommand{\startContent}{
    \pagestyle{plain}
    \clearpage
    \pagenumbering{arabic} 
}




\newenvironment{ToC}[1][]{
    \unNumChapter{Table of Contents}
   \begin{spacing}{1.5}
}{
   \end{spacing}   
    \newpage
}




\newif\iffigures
\newif\iftables
\newif\ifabbrs
\newif\ifappens

\makeatletter
\AtEndDocument {%
    \if@filesw
       \ifnum\value{figureCount}=\z@ % no figures
          \immediate\write\@mainaux {\global\string\figuresfalse}%
       \else
          \immediate\write\@mainaux {\global\string\figurestrue}%
       \fi
       \ifnum\value{tableCount}=\z@ % no tables
          \immediate\write\@mainaux {\global\string\tablesfalse}%
       \else
          \immediate\write\@mainaux {\global\string\tablestrue}%
       \fi
       \ifnum\value{abbrCount}=\z@ % no Abbreviations
          \immediate\write\@mainaux {\global\string\abbrsfalse}%
       \else
          \immediate\write\@mainaux {\global\string\abbrstrue}%
       \fi
       \ifnum\value{appenCount}=\z@ % no Appendices
          \immediate\write\@mainaux {\global\string\appensfalse}%
       \else
          \immediate\write\@mainaux {\global\string\appenstrue}%
       \fi
    \fi
}
\makeatother 


\newenvironment{LoF}[1][]{
\unNumChapter{List of Figures}
\textbf{Figure}\hspace{1.3cm}\textbf{Description}\hspace{9.7cm}\textbf{Page}\\
}{}

\newenvironment{LoT}[1][]{
\unNumChapter{List of Tables}
\textbf{Table}\hspace{1.5cm}\textbf{Description}\hspace{9.75cm}\textbf{Page}\\
}{}

\newenvironment{LoAp}[1][]{
\unNumChapter{List of Appendices}
\textbf{Appendix}\hspace{1.5cm}\textbf{Description}\hspace{8.7cm}\textbf{Page}\\
}{}



\newcommand{\styleSwitch}[2]{\ifapaT #1 \else #2 \fi}



\newenvironment{references}[1][]{\renewcommand{\bibname}{}
\unNumChapter{References}
\vspace{-2cm}
\ifapaT  %APA
    %\bibliographystyle{unsrtnat}
    \bibliographystyle{apalike}
\else % IEEE
    \bibliographystyle{IEEEtranN}
\fi
}{}