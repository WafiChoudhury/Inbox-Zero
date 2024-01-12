// HEADER ANIMATION
window.onscroll = function () { scrollFunction() };
var element = document.getElementById("body");
function scrollFunction() {
  if (document.body.scrollTop > 400 || document.documentElement.scrollTop > 400) {
    $(".navbar").addClass("fixed-top");
    element.classList.add("header-small");
    $("body").addClass("body-top-padding");

  } else {
    $(".navbar").removeClass("fixed-top");
    element.classList.remove("header-small");
    $("body").removeClass("body-top-padding");
  }
}


// document.getElementById("submitBtn").addEventListener("click", function () {
//   // Get form data
//   const token = document.getElementById("token").value;

//   // Construct the data to be sent to the Flask backend
//   const formData = {
//     token: token
//   };
//   // Send data to Flask endpoint using fetch
//   fetch('/', {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify(formData)
//   })
//     .then(response => {
//       // Handle the response from the Flask backend
//       // For example, you can show a success message or handle errors
//       console.log('Data sent to Flask endpoint');
//     })
//     .catch(error => {
//       // Handle errors if the request fails
//       console.error('Error:', error);
//     });
// });


document.addEventListener("DOMContentLoaded", function () {
  // Find the "Get started" button by its ID
  const getStartedButton = document.getElementById("getStartedBtn");

  // Add a click event listener to the button
  getStartedButton.addEventListener("click", function () {
    // Redirect to a different route when the button is clicked
    window.location.href = "/canvas"; // Replace "/different-route" with your desired route
  });
});


// document.addEventListener("DOMContentLoaded", function () {
//   // Find the "Get started" button by its ID
//   const getStartedButton = document.getElementById("submitBtn");

//   // Add a click event listener to the button
//   getStartedButton.addEventListener("click", function () {
//     // Redirect to a different route when the button is clicked
//     window.location.href = "/"; // Replace "/different-route" with your desired route
//   });
// });

// OWL-CAROUSAL
$('.owl-carousel').owlCarousel({
  items: 3,
  loop: true,
  nav: false,
  dot: true,
  autoplay: true,
  slideTransition: 'linear',
  autoplayHoverPause: true,
  responsive: {
    0: {
      items: 1
    },
    600: {
      items: 2
    },
    1000: {
      items: 3
    }
  }
})

// SCROLLSPY
$(document).ready(function () {
  $(".nav-link").click(function () {
    var t = $(this).attr("href");
    $("html, body").animate({
      scrollTop: $(t).offset().top - 75
    }, {
      duration: 1000,
    });
    $('body').scrollspy({ target: '.navbar', offset: $(t).offset().top });
    return false;
  });

});

// AOS
AOS.init({
  offset: 120,
  delay: 0,
  duration: 1200,
  easing: 'ease',
  once: true,
  mirror: false,
  anchorPlacement: 'top-bottom',
  disable: "mobile"
});

//SIDEBAR-OPEN
$('#navbarSupportedContent').on('hidden.bs.collapse', function () {
  $("body").removeClass("sidebar-open");
})

$('#navbarSupportedContent').on('shown.bs.collapse', function () {
  $("body").addClass("sidebar-open");
})


window.onresize = function () {
  var w = window.innerWidth;
  if (w >= 992) {
    $('body').removeClass('sidebar-open');
    $('#navbarSupportedContent').removeClass('show');
  }
}